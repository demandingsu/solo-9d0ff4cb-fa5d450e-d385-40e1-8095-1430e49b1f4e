"""Tile validation, effective-tile selection and deterministic mosaic composition."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


def quote_field(value: str) -> str:
    """Minimal RFC 4180 quoting for the two free-text CSV columns."""
    if any(ch in value for ch in ',"\r\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


TILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
MAX_TILE_ID_LEN = 64


class TileValidationError(ValueError):
    """Raised when an uploaded tile or canvas setting is invalid."""


@dataclass(frozen=True)
class TileRecord:
    id: int
    tile_id: str
    revision: int
    x: int
    y: int
    width: int
    height: int
    image_rel: str
    mask_rel: str
    image_sha256: str
    mask_sha256: str
    cloud_pixels: int
    clear_pixels: int
    submitted_at: str
    status: str = "active"

    @property
    def sort_key(self) -> tuple[datetime, int]:
        return (parse_timestamp(self.submitted_at), self.id)


def parse_timestamp(value: str) -> datetime:
    """Parse our stored ISO-8601 timestamps as timezone-aware UTC datetimes."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def validate_tile_id(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        raise TileValidationError("瓦片号不能为空。")
    if len(value) > MAX_TILE_ID_LEN:
        raise TileValidationError("瓦片号最多 64 个字符。")
    if not TILE_ID_RE.match(value):
        raise TileValidationError(
            "瓦片号只能包含字母、数字、下划线、点和连字符，且须以字母或数字开头。"
        )
    return value


def validate_revision(raw: str | int) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise TileValidationError("修订号必须是整数。") from None
    if value < 1:
        raise TileValidationError("修订号必须是正整数（从 1 开始）。")
    return value


def validate_int(raw: str | int, label: str) -> int:
    try:
        text = str(raw).strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            raise ValueError
        return int(text)
    except (TypeError, ValueError):
        raise TileValidationError(f"{label}必须是整数。") from None


def validate_canvas(width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise TileValidationError("画幅宽高必须为正整数。")
    if width > 20000 or height > 20000:
        raise TileValidationError("画幅宽高不能超过 20000 像素。")
    if width * height > 16_000_000:
        raise TileValidationError("画幅像素总量不能超过 1600 万（逐像素来源 CSV 会很庞大）。")


@dataclass
class ValidatedTile:
    tile_id: str
    revision: int
    x: int
    y: int
    width: int
    height: int
    image_bytes: bytes
    mask_bytes: bytes
    image_sha256: str
    mask_sha256: str
    cloud_pixels: int
    clear_pixels: int


def _decode_png(raw: bytes, label: str) -> Image.Image:
    if not raw:
        raise TileValidationError(f"{label}为空。")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        raise TileValidationError(f"{label}不是可读取的 PNG 图像。") from None
    if image.format != "PNG":
        raise TileValidationError(f"{label}必须是 PNG 格式。")
    return image


def validate_upload(
    *,
    tile_id: str,
    revision: str | int,
    x: str | int,
    y: str | int,
    image_bytes: bytes,
    mask_bytes: bytes,
) -> ValidatedTile:
    """Validate one RGB image plus same-size binary cloud mask.

    Cloud pixels are marks with any non-zero mask value.  The mask is
    interpreted as fully opaque cloud coverage: a cloudy pixel may never be
    used as a colour source.
    """
    tile_id = validate_tile_id(tile_id)
    revision = validate_revision(revision)
    x = validate_int(x, "X 位置")
    y = validate_int(y, "Y 位置")

    image = _decode_png(image_bytes, "瓦片图像")
    mask = _decode_png(mask_bytes, "云掩膜")

    if image.mode != "RGB":
        raise TileValidationError(f"瓦片图像必须是 RGB PNG（当前为 {image.mode}）。")
    if mask.mode not in ("1", "L", "P"):
        raise TileValidationError(
            f"云掩膜必须是同样尺寸的二值 PNG（当前为 {mask.mode}）。"
        )
    if image.size != mask.size:
        raise TileValidationError(
            "云掩膜尺寸必须与瓦片图像一致"
            f"（图像 {image.width}x{image.height}，掩膜 {mask.width}x{mask.height}）。"
        )

    width, height = image.size
    if width < 1 or height < 1:
        raise TileValidationError("瓦片尺寸必须为正。")

    if mask.mode != "L":
        mask = mask.convert("L")
    histogram = mask.histogram()
    clear_pixels = int(histogram[0])
    cloud_pixels = width * height - clear_pixels

    return ValidatedTile(
        tile_id=tile_id,
        revision=revision,
        x=x,
        y=y,
        width=width,
        height=height,
        image_bytes=image_bytes,
        mask_bytes=mask_bytes,
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
        cloud_pixels=cloud_pixels,
        clear_pixels=clear_pixels,
    )


def effective_tiles(records: Iterable[TileRecord]) -> list[TileRecord]:
    """Newest active revision per tile id, ordered by submission time."""
    newest: dict[str, TileRecord] = {}
    for record in records:
        if record.status != "active":
            continue
        current = newest.get(record.tile_id)
        if current is None or (record.revision, record.sort_key) > (
            current.revision,
            current.sort_key,
        ):
            newest[record.tile_id] = record
    return sorted(newest.values(), key=lambda r: r.sort_key)


def compute_fingerprint(
    canvas_width: int,
    canvas_height: int,
    tiles: Sequence[TileRecord],
) -> str:
    """Stable digest of the exact inputs a publish preview was built from."""
    digest = hashlib.sha256()
    digest.update(b"mosaic-preview-v1\0")
    digest.update(f"{canvas_width}x{canvas_height}\0".encode())
    for tile in sorted(tiles, key=lambda t: (t.tile_id, t.revision, t.id)):
        digest.update(
            (
                f"{tile.id}\0{tile.tile_id}\0{tile.revision}\0{tile.x}\0{tile.y}\0"
                f"{tile.width}\0{tile.height}\0{tile.image_sha256}\0"
                f"{tile.mask_sha256}\0"
            ).encode()
        )
    return digest.hexdigest()


@dataclass
class _Crop:
    tile: TileRecord
    x0: int
    y0: int
    x1: int
    y1: int
    rank: int
    clear: bool = False

    @property
    def placement(self) -> tuple[int, int]:
        return (self.x0 - self.tile.x, self.y0 - self.tile.y)


@dataclass
class RenderStats:
    width: int
    height: int
    covered_pixels: int
    transparent_pixels: int


def _flat_pixels(image: Image.Image):
    """Pillow 13 renamed getdata(); support both old and new releases."""
    getter = getattr(image, "get_flattened_data", None)
    if getter is not None:
        return getter()
    return image.getdata()


def render_mosaic(
    canvas_width: int,
    canvas_height: int,
    tiles: Sequence[TileRecord],
    storage_root: Path,
    png_path: Path,
    csv_path: Path,
) -> RenderStats:
    """Compose the canvas and provenance CSV into the given files.

    Cloud pixels are transparent.  For the remaining pixels the most recently
    submitted tile (by submission timestamp, not revision number) wins, as
    long as it has a cloud-free value there; otherwise the earlier tiles act
    as fallbacks.  Pixels with no clear source stay transparent.
    """
    validate_canvas(canvas_width, canvas_height)

    # Newest first: index 0 has the highest priority.
    ordered = sorted(tiles, key=lambda t: t.sort_key, reverse=True)
    crops: list[_Crop] = []
    for rank, tile in enumerate(ordered):
        x0 = max(tile.x, 0)
        y0 = max(tile.y, 0)
        x1 = min(tile.x + tile.width, canvas_width)
        y1 = min(tile.y + tile.height, canvas_height)
        if x1 <= x0 or y1 <= y0:
            continue
        crops.append(_Crop(tile=tile, x0=x0, y0=y0, x1=x1, y1=y1, rank=rank))

    canvas = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
    alpha = Image.new("L", (canvas_width, canvas_height), 0)
    owner = Image.new("I", (canvas_width, canvas_height), 0xFFFFFFFF)
    sentinel = 0xFFFFFFFF

    # Apply oldest first so newer tiles paint over them.
    for crop in reversed(crops):
        tile_path = storage_root / crop.tile.image_rel
        mask_path = storage_root / crop.tile.mask_rel
        with Image.open(tile_path) as image, Image.open(mask_path) as mask:
            image = image.convert("RGB")
            mask = mask.convert("L")
            if image.size != (crop.tile.width, crop.tile.height):
                raise RuntimeError(
                    f"瓦片 {crop.tile.tile_id} r{crop.tile.revision} 文件尺寸已变化。"
                )
            left, top = crop.placement
            box = (left, top, left + (crop.x1 - crop.x0), top + (crop.y1 - crop.y0))
            image_crop = image.crop(box)
            mask_crop = mask.crop(box)
        clear_mask = mask_crop.point(lambda v: 255 if v == 0 else 0)
        mask_pixels = list(_flat_pixels(clear_mask))
        if not any(mask_pixels):
            continue
        crop.clear = True
        canvas.paste(image_crop, (crop.x0, crop.y0), clear_mask)
        alpha.paste(255, (crop.x0, crop.y0), clear_mask)
        # Stream the owner map row by row (rank + 1, so 0 stays "no source").
        crop_width = crop.x1 - crop.x0
        overlay = Image.new("I", (crop_width, crop.y1 - crop.y0), 0)
        overlay_rows = []
        for start in range(0, len(mask_pixels), crop_width):
            overlay_rows.extend(
                crop.rank + 1 if value else 0
                for value in mask_pixels[start:start + crop_width]
            )
        overlay.putdata(overlay_rows)
        owner.paste(overlay, (crop.x0, crop.y0), clear_mask)

    rgba = canvas.convert("RGBA")
    rgba.putalpha(alpha)
    rgba.save(png_path, format="PNG")

    rank_to_tile = {crop.rank: crop.tile for crop in crops if crop.clear}
    alphas = _flat_pixels(alpha)
    covered = 0
    label_for_rank = {
        rank: (
            quote_field(tile.tile_id),
            str(tile.revision),
        )
        for rank, tile in rank_to_tile.items()
    }
    blank = ("", "")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("x,y,tile_id,revision\n")
        rows = []
        y = 0
        x = 0
        owners = _flat_pixels(owner)
        for index, (rank_marker, opacity) in enumerate(zip(owners, alphas)):
            if opacity == 0 or rank_marker == sentinel:
                label = blank
            else:
                covered += 1
                label = label_for_rank.get(rank_marker - 1, blank)
            rows.append(f"{x},{y},{label[0]},{label[1]}")
            x += 1
            if x == canvas_width:
                x = 0
                y += 1
            if len(rows) >= 20000:
                handle.write("\n".join(rows) + "\n")
                rows.clear()
        if rows:
            handle.write("\n".join(rows) + "\n")

    total = canvas_width * canvas_height
    return RenderStats(
        width=canvas_width,
        height=canvas_height,
        covered_pixels=covered,
        transparent_pixels=total - covered,
    )

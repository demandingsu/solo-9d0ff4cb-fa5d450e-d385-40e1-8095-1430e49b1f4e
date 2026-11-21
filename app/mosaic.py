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
class RenderStats:
    width: int
    height: int
    covered_pixels: int
    transparent_pixels: int


def _decode_png_file(path: Path, mode: str) -> Image.Image:
    """Decode one PNG fully into memory in the wanted mode.

    The file handle is closed before returning, so the decoded buffer is the
    only allocation kept by the caller.
    """
    with Image.open(path) as raw:
        raw.load()
        return raw if raw.mode == mode else raw.convert(mode)


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

    The pipeline is deliberately memory-frugal so that a full-size canvas
    (1600 万像素) composited from multi-megapixel tiles fits in a few hundred
    MiB: tiles are pasted straight from their decoded buffers (Pillow clips
    the off-canvas part, so no crop copies are made), the owner map uses the
    narrowest pixel mode that can hold the tile ranks, and the CSV is
    streamed row by row instead of building per-pixel Python lists.
    """
    validate_canvas(canvas_width, canvas_height)

    # Newest first: index 0 has the highest priority.  Tiles falling fully
    # outside the canvas are dropped; the survivors are re-ranked compactly
    # (still newest first) so the owner map below stays as narrow as possible.
    ordered = sorted(tiles, key=lambda t: t.sort_key, reverse=True)
    ranked: list[tuple[int, TileRecord]] = []
    for tile in ordered:
        if (
            max(tile.x, 0) < min(tile.x + tile.width, canvas_width)
            and max(tile.y, 0) < min(tile.y + tile.height, canvas_height)
        ):
            ranked.append((len(ranked), tile))

    # The owner map records which tile (rank + 1) supplied each pixel; 0 means
    # "no source".  Pick the narrowest mode that can hold the ranks so the map
    # costs 1/2/4 bytes per pixel instead of always 4.  `owner_cast` is the
    # memoryview format matching the mode's tobytes() layout.
    if len(ranked) <= 255:
        owner_mode, owner_cast = "L", "B"
    elif len(ranked) <= 65535:
        owner_mode, owner_cast = "I;16", "H"
    else:
        owner_mode, owner_cast = "I", "I"

    canvas = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
    alpha = Image.new("L", (canvas_width, canvas_height), 0)
    owner = Image.new(owner_mode, (canvas_width, canvas_height), 0)

    used_ranks: set[int] = set()

    # Apply oldest first so newer tiles paint over them.  The mask is decoded
    # before the (much larger) image so fully cloudy tiles never pay for the
    # image decode, and so the mask buffer is freed before the image loads.
    for rank, tile in reversed(ranked):
        expected_size = (tile.width, tile.height)
        mask = _decode_png_file(storage_root / tile.mask_rel, "L")
        if mask.size != expected_size:
            raise RuntimeError(
                f"瓦片 {tile.tile_id} r{tile.revision} 文件尺寸已变化。"
            )
        # 255 where the tile is cloud-free, 0 where it is cloudy.
        clear_mask = mask.point(lambda v: 255 if v == 0 else 0)
        del mask
        if clear_mask.getextrema()[1] == 0:
            del clear_mask
            continue  # fully cloudy tile contributes nothing
        image = _decode_png_file(storage_root / tile.image_rel, "RGB")
        if image.size != expected_size:
            raise RuntimeError(
                f"瓦片 {tile.tile_id} r{tile.revision} 文件尺寸已变化。"
            )
        used_ranks.add(rank)
        # Paste at the tile's canvas position; Pillow clips the off-canvas
        # part, so no cropped copies of the tile are needed.
        position = (tile.x, tile.y)
        canvas.paste(image, position, clear_mask)
        alpha.paste(255, position, clear_mask)
        owner.paste(rank + 1, position, clear_mask)
        del image, clear_mask

    total = canvas_width * canvas_height
    transparent = alpha.histogram()[0]
    covered = total - transparent

    rgba = canvas.convert("RGBA")
    del canvas
    rgba.putalpha(alpha)
    del alpha
    rgba.save(png_path, format="PNG")
    del rgba

    # Per-pixel provenance CSV (one line per pixel, top-to-bottom then
    # left-to-right), streamed one canvas row at a time.  `suffix[0]` (no
    # source) renders as two empty fields, exactly like the labelled entries.
    suffix = [",,"] * (len(ranked) + 1)
    for rank, tile in ranked:
        if rank in used_ranks:
            suffix[rank + 1] = f",{quote_field(tile.tile_id)},{tile.revision}"
    owners: bytes | memoryview = owner.tobytes()
    del owner
    if owner_cast != "B":
        owners = memoryview(owners).cast(owner_cast)
    x_parts = [f"{x}," for x in range(canvas_width)]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("x,y,tile_id,revision\n")
        write = handle.write
        index = 0
        for y in range(canvas_height):
            ys = str(y)
            prefixes = [x_part + ys for x_part in x_parts]
            lines: list[str] = []
            append = lines.append
            for x in range(canvas_width):
                append(prefixes[x] + suffix[owners[index]])
                index += 1
            write("\n".join(lines))
            write("\n")

    return RenderStats(
        width=canvas_width,
        height=canvas_height,
        covered_pixels=covered,
        transparent_pixels=transparent,
    )

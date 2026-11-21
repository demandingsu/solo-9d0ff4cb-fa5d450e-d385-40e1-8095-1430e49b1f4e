"""FastAPI web application for the cloud-free aerial mosaic."""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import threading
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import db
from .mosaic import (
    MAX_TILE_ID_LEN,
    TileRecord,
    TileValidationError,
    compute_fingerprint,
    effective_tiles,
    render_mosaic,
    utc_now_iso,
    validate_canvas,
    validate_int,
    validate_upload,
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Files in an uncommitted preview directory are named with this prefix while
# they are being written.  Downloads never serve a directory containing one.
STAGING_PREFIX = ".tmp-"
PNG_NAME = "mosaic.png"
CSV_NAME = "sources.csv"
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_TILE_PIXELS = 50_000_000


# --------------------------------------------------------------------------- #
# Application state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Runtime:
    data_dir: Path
    db_path: Path
    storage_dir: Path
    previews_dir: Path
    versions_dir: Path

    @classmethod
    def build(cls, root: Path) -> "Runtime":
        root = Path(root).resolve()
        previews = root / "previews"
        versions = root / "published"
        storage = root / "tiles"
        for directory in (root, previews, versions, storage):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(
            data_dir=root,
            db_path=root / "mosaic.sqlite3",
            storage_dir=storage,
            previews_dir=previews,
            versions_dir=versions,
        )


class LockedConnection:
    """Serialised access to one SQLite connection and the filesystem state."""

    def __init__(self, db_path: Path) -> None:
        self._lock = threading.RLock()
        self._conn = db.connect(db_path)

    def __enter__(self):
        self._lock.acquire()
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._lock.release()
        return False

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def create_app(data_dir: str | os.PathLike[str] | None = None) -> FastAPI:
    root = Path(data_dir or os.environ.get("MOSAIC_DATA_DIR", "./data"))
    runtime = Runtime.build(root)
    db.init_db(runtime.db_path)
    db_lock = LockedConnection(runtime.db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            db_lock.close()

    app = FastAPI(title="航拍无云画幅合成", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.db_lock = db_lock
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #

    def load_tiles(conn) -> list[TileRecord]:
        rows = conn.execute(
            "SELECT id, tile_id, revision, x, y, width, height, image_rel, mask_rel, "
            "image_sha256, mask_sha256, cloud_pixels, clear_pixels, submitted_at, status "
            "FROM revisions ORDER BY id"
        ).fetchall()
        return [
            TileRecord(
                id=row["id"],
                tile_id=row["tile_id"],
                revision=row["revision"],
                x=row["x"],
                y=row["y"],
                width=row["width"],
                height=row["height"],
                image_rel=row["image_rel"],
                mask_rel=row["mask_rel"],
                image_sha256=row["image_sha256"],
                mask_sha256=row["mask_sha256"],
                cloud_pixels=row["cloud_pixels"],
                clear_pixels=row["clear_pixels"],
                submitted_at=row["submitted_at"],
                status=row["status"],
            )
            for row in rows
        ]

    def preview_dir_for(fingerprint: str) -> Path:
        return runtime.previews_dir / fingerprint

    def read_preview_state(conn) -> dict[str, Any]:
        width = db.get_int_setting(conn, "canvas_width", 512)
        height = db.get_int_setting(conn, "canvas_height", 512)
        return {
            "width": width,
            "height": height,
            "version": db.get_int_setting(conn, "preview_version", -1),
            "state_version": db.get_int_setting(conn, "state_version", 0),
            "error": db.get_setting(conn, "preview_error", ""),
            "covered": db.get_int_setting(conn, "draft_covered", 0),
            "transparent": db.get_int_setting(conn, "draft_transparent", 0),
        }

    def mark_state_changed(conn) -> int:
        version = db.get_int_setting(conn, "state_version", 0) + 1
        db.set_setting(conn, "state_version", version)
        return version

    def publish_preview() -> bool:
        """Render the current inputs into a content-addressed preview dir."""
        try:
            with db_lock as conn:
                width = db.get_int_setting(conn, "canvas_width", 512)
                height = db.get_int_setting(conn, "canvas_height", 512)
                state_version = db.get_int_setting(conn, "state_version", 0)
                current = effective_tiles(load_tiles(conn))
                fingerprint = compute_fingerprint(width, height, current)
                target = preview_dir_for(fingerprint)
                staging = runtime.previews_dir / f"{STAGING_PREFIX}{secrets.token_hex(12)}"
                staging.mkdir(parents=True, exist_ok=False)

            # Render outside the lock so uploads/withdrawals stay responsive.
            try:
                stats = render_mosaic(
                    width, height, current, runtime.storage_dir,
                    staging / PNG_NAME, staging / CSV_NAME,
                )
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            if target.is_dir():
                shutil.rmtree(staging, ignore_errors=True)
            else:
                try:
                    os.replace(staging, target)
                except OSError:
                    # A concurrent refresh won the race; both renders are of
                    # the same fingerprint, so discarding ours is correct.
                    shutil.rmtree(staging, ignore_errors=True)

            with db_lock as conn:
                # Only claim "current" if the draft did not move on while we
                # were rendering; otherwise the caller's newer change is the
                # one that will refresh the preview.
                still_current = (
                    db.get_int_setting(conn, "state_version", 0) == state_version
                    and compute_fingerprint(
                        width,
                        height,
                        effective_tiles(load_tiles(conn)),
                    )
                    == fingerprint
                )
                if still_current:
                    db.set_setting(conn, "preview_version", state_version)
                    db.set_setting(conn, "draft_covered", stats.covered_pixels)
                    db.set_setting(conn, "draft_transparent", stats.transparent_pixels)
                db.set_setting(conn, "preview_error", "")
                conn.commit()
            return True
        except Exception:
            detail = traceback.format_exc(limit=4)
            print("preview refresh failed:\n" + detail)
            try:
                with db_lock as conn:
                    db.set_setting(
                        conn,
                        "preview_error",
                        "预览或发布生成失败；已保留上一完整版本。请修正输入后重试。",
                    )
                    conn.commit()
            except Exception:
                pass
            return False

    # ------------------------------------------------------------------ #
    # View helpers
    # ------------------------------------------------------------------ #

    def build_context(request: Request, conn) -> dict[str, Any]:
        state = read_preview_state(conn)
        tiles = load_tiles(conn)
        active = [t for t in tiles if t.status == "active"]
        withdrawn = [t for t in tiles if t.status == "withdrawn"]
        current = effective_tiles(tiles)
        width, height = state["width"], state["height"]
        fingerprint = compute_fingerprint(width, height, current)
        preview_ready = (
            state["version"] == state["state_version"]
            and (preview_dir_for(fingerprint) / PNG_NAME).is_file()
        )
        published = conn.execute(
            "SELECT * FROM published_versions ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        published_version = None
        if published is not None:
            published_version = {
                "id": published["id"],
                "seq": published["seq"],
                "created_at": published["created_at"],
                "fingerprint": published["fingerprint"],
                "width": published["canvas_width"],
                "height": published["canvas_height"],
                "tile_count": published["tile_count"],
                "covered": published["covered_pixels"],
                "transparent": published["transparent_pixels"],
                "png_bytes": published["png_bytes"],
                "csv_bytes": published["csv_bytes"],
                "stale": published["fingerprint"] != fingerprint
                or published["canvas_width"] != width
                or published["canvas_height"] != height,
            }
        recent = [
            {
                "id": row["id"],
                "seq": row["seq"],
                "created_at": row["created_at"],
                "tile_count": row["tile_count"],
                "width": row["canvas_width"],
                "height": row["canvas_height"],
            }
            for row in conn.execute(
                "SELECT * FROM published_versions ORDER BY seq DESC LIMIT 10"
            ).fetchall()
        ]
        return {
            "request": request,
            "state": state,
            "preview_ready": preview_ready,
            "current": current,
            "current_ids": {tile.id for tile in current},
            "active": sorted(active, key=lambda t: (t.tile_id, -t.revision)),
            "withdrawn": sorted(withdrawn, key=lambda t: -t.id),
            "tiles": tiles,
            "published": published_version,
            "recent_versions": recent,
            "fingerprint": fingerprint,
            "limits": {"max_tile_id_len": MAX_TILE_ID_LEN},
            "flash": request.query_params.get("msg", ""),
            "flash_error": request.query_params.get("err", ""),
        }

    def render_page(request: Request, status_code: int = 200) -> HTMLResponse:
        with db_lock as conn:
            context = build_context(request, conn)
        return templates.TemplateResponse(
            request, "index.html", context, status_code=status_code
        )

    def redirect(msg: str | None = None, err: str | None = None) -> RedirectResponse:
        params = []
        if msg:
            params.append("msg=" + quote(msg))
        if err:
            params.append("err=" + quote(err))
        url = "/" + ("?" + "&".join(params) if params else "")
        return RedirectResponse(url, status_code=303)

    # ------------------------------------------------------------------ #
    # Routes
    # ------------------------------------------------------------------ #

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return render_page(request)

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.post("/canvas")
    def set_canvas(
        width: str = Form(...),
        height: str = Form(...),
    ) -> RedirectResponse:
        try:
            w = validate_int(width, "画幅宽度")
            h = validate_int(height, "画幅高度")
            validate_canvas(w, h)
        except TileValidationError as exc:
            return redirect(err=str(exc))
        with db_lock as conn:
            db.set_setting(conn, "canvas_width", w)
            db.set_setting(conn, "canvas_height", h)
            mark_state_changed(conn)
        if not publish_preview():
            return redirect(err="画幅已更新，但预览生成失败。")
        return redirect(msg=f"画幅已设为 {w}x{h}。")

    @app.post("/tiles")
    async def upload_tile(
        tile_id: str = Form(...),
        revision: str = Form(...),
        x: str = Form(...),
        y: str = Form(...),
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
    ) -> RedirectResponse:
        image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
        mask_bytes = await mask.read(MAX_UPLOAD_BYTES + 1)
        if len(image_bytes) > MAX_UPLOAD_BYTES or len(mask_bytes) > MAX_UPLOAD_BYTES:
            return redirect(err="单个上传文件不能超过 64 MB。")
        try:
            checked = validate_upload(
                tile_id=tile_id,
                revision=revision,
                x=x,
                y=y,
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
            )
        except TileValidationError as exc:
            return redirect(err=str(exc))
        if checked.width * checked.height > MAX_TILE_PIXELS:
            return redirect(err="单个瓦片不能超过 5000 万像素。")

        with db_lock as conn:
            existing = conn.execute(
                "SELECT id, status FROM revisions WHERE tile_id = ? AND revision = ?",
                (checked.tile_id, checked.revision),
            ).fetchall()
            if existing:
                if any(row["status"] == "active" for row in existing):
                    return redirect(
                        err=f"瓦片 {checked.tile_id} 修订 r{checked.revision} 已存在。"
                    )
                return redirect(
                    err=f"瓦片 {checked.tile_id} 修订 r{checked.revision} 已撤回，"
                    "请用更高的修订号重新提交。"
                )

            token = secrets.token_hex(16)
            subdir = runtime.storage_dir / checked.tile_id / f"r{checked.revision}"
            subdir.mkdir(parents=True, exist_ok=True)
            image_name = f"{token}.png"
            mask_name = f"{token}.mask.png"
            staging_dir = subdir / f"{STAGING_PREFIX}{token}"
            staging_dir.mkdir(parents=True, exist_ok=True)
            (staging_dir / image_name).write_bytes(checked.image_bytes)
            (staging_dir / mask_name).write_bytes(checked.mask_bytes)
            os.replace(staging_dir / image_name, subdir / image_name)
            os.replace(staging_dir / mask_name, subdir / mask_name)
            shutil.rmtree(staging_dir, ignore_errors=True)

            image_rel = str(
                (subdir / image_name).relative_to(runtime.storage_dir)
            )
            mask_rel = str(
                (subdir / mask_name).relative_to(runtime.storage_dir)
            )
            try:
                conn.execute(
                    "INSERT INTO revisions(tile_id, revision, x, y, width, height, "
                    "image_rel, mask_rel, image_sha256, mask_sha256, cloud_pixels, "
                    "clear_pixels, submitted_at, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')",
                    (
                        checked.tile_id,
                        checked.revision,
                        checked.x,
                        checked.y,
                        checked.width,
                        checked.height,
                        image_rel,
                        mask_rel,
                        checked.image_sha256,
                        checked.mask_sha256,
                        checked.cloud_pixels,
                        checked.clear_pixels,
                        utc_now_iso(),
                    ),
                )
            except Exception:
                shutil.rmtree(subdir, ignore_errors=True)
                raise
            mark_state_changed(conn)

        if not publish_preview():
            return redirect(err="瓦片已保存，但预览生成失败。")
        return redirect(
            msg=f"已提交瓦片 {checked.tile_id} 修订 r{checked.revision}。"
        )

    @app.post("/tiles/{revision_id}/withdraw")
    def withdraw_tile(revision_id: int) -> RedirectResponse:
        with db_lock as conn:
            row = conn.execute(
                "SELECT * FROM revisions WHERE id = ?", (revision_id,)
            ).fetchone()
            if row is None:
                return redirect(err="未找到该瓦片修订。")
            if row["status"] != "active":
                return redirect(err="该瓦片修订已经撤回。")
            newer = conn.execute(
                "SELECT 1 FROM revisions WHERE tile_id = ? AND revision > ? "
                "AND status = 'active' LIMIT 1",
                (row["tile_id"], row["revision"]),
            ).fetchone()
            conn.execute(
                "UPDATE revisions SET status = 'withdrawn', withdrawn_at = ? "
                "WHERE id = ?",
                (utc_now_iso(), revision_id),
            )
            mark_state_changed(conn)
        if not publish_preview():
            return redirect(err="瓦片已撤回，但预览重算失败。")
        note = ""
        if newer:
            note = "（存在更高修订号，实际采用更高修订。）"
        return redirect(
            msg=f"已撤回瓦片 {row['tile_id']} 修订 r{row['revision']}{note}。"
        )

    @app.post("/publish")
    def publish() -> RedirectResponse:
        # Make sure the preview is current before freezing it.
        with db_lock as conn:
            state = read_preview_state(conn)
        preview_current = state["version"] == state["state_version"] and not state["error"]
        if not preview_current:
            if not publish_preview():
                return redirect(err="预览不可用，发布失败；上一版本仍可下载。")
            with db_lock as conn:
                state = read_preview_state(conn)
            if state["version"] != state["state_version"]:
                return redirect(
                    err="草稿在发布前仍在变化，请稍后重试发布；上一版本仍可下载。"
                )
        try:
            version = _publish_snapshot()
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            print("publish failed:\n" + traceback.format_exc(limit=5))
            reason = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else "未知错误"
            return redirect(err=f"发布失败：{reason}。上一完整版本仍可下载。")
        return redirect(
            msg=f"已发布版本 #{version['seq']}（{version['tile_count']} 个瓦片）。"
        )

    def _publish_snapshot() -> dict[str, Any]:
        # Phase 1: capture the exact inputs under the state lock.
        with db_lock as conn:
            width = db.get_int_setting(conn, "canvas_width", 512)
            height = db.get_int_setting(conn, "canvas_height", 512)
            tiles = load_tiles(conn)
            current = effective_tiles(tiles)
            fingerprint = compute_fingerprint(width, height, current)
            seq = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM published_versions"
            ).fetchone()[0]

        # Phase 2: render the frozen inputs into a private staging directory.
        # Concurrent edits only change the database, never `current`, so this
        # snapshot cannot mix inputs from two different drafts.
        version_id = secrets.token_hex(12)
        dir_name = f"{seq:06d}-{version_id}"
        final_dir = runtime.versions_dir / dir_name
        staging = runtime.versions_dir / f"{STAGING_PREFIX}{version_id}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            stats = render_mosaic(
                width, height, current, runtime.storage_dir,
                staging / PNG_NAME, staging / CSV_NAME,
            )
            png_sha = hashlib.sha256((staging / PNG_NAME).read_bytes()).hexdigest()
            csv_sha = hashlib.sha256((staging / CSV_NAME).read_bytes()).hexdigest()
            (staging / "manifest.json").write_text(
                _manifest_json(
                    version_id,
                    fingerprint,
                    width,
                    height,
                    current,
                    png_sha,
                    csv_sha,
                ),
                encoding="utf-8",
            )
            # Atomic publish: the directory only becomes visible complete.
            os.replace(staging, final_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        png_bytes = (final_dir / PNG_NAME).stat().st_size
        csv_bytes = (final_dir / CSV_NAME).stat().st_size
        covered = stats.covered_pixels
        try:
            with db_lock as conn:
                conn.execute(
                    "INSERT INTO published_versions(id, seq, dir_rel, created_at, "
                    "fingerprint, canvas_width, canvas_height, tile_count, "
                    "covered_pixels, transparent_pixels, png_rel, csv_rel, "
                    "png_sha256, csv_sha256, png_bytes, csv_bytes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        version_id,
                        seq,
                        dir_name,
                        utc_now_iso(),
                        fingerprint,
                        width,
                        height,
                        len(current),
                        covered,
                        width * height - covered,
                        f"{dir_name}/{PNG_NAME}",
                        f"{dir_name}/{CSV_NAME}",
                        png_sha,
                        csv_sha,
                        png_bytes,
                        csv_bytes,
                    ),
                )
                db.set_setting(conn, "published_version_id", version_id)
        except Exception:
            shutil.rmtree(final_dir, ignore_errors=True)
            raise
        return {"id": version_id, "seq": seq, "tile_count": len(current)}

    def _manifest_json(
        version_id: str,
        fingerprint: str,
        width: int,
        height: int,
        tiles: list[TileRecord],
        png_sha: str,
        csv_sha: str,
    ) -> str:
        import json

        return json.dumps(
            {
                "version_id": version_id,
                "fingerprint": fingerprint,
                "canvas": {"width": width, "height": height},
                "png_sha256": png_sha,
                "csv_sha256": csv_sha,
                "tiles": [
                    {
                        "tile_id": tile.tile_id,
                        "revision": tile.revision,
                        "x": tile.x,
                        "y": tile.y,
                        "width": tile.width,
                        "height": tile.height,
                        "image_sha256": tile.image_sha256,
                        "mask_sha256": tile.mask_sha256,
                    }
                    for tile in sorted(tiles, key=lambda t: (t.tile_id, t.revision))
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------ #
    # Preview
    # ------------------------------------------------------------------ #

    @app.get("/preview.png")
    def preview_png() -> FileResponse:
        with db_lock as conn:
            state = read_preview_state(conn)
            if state["version"] != state["state_version"]:
                raise HTTPException(status_code=409, detail="预览尚未生成完成。")
            width, height = state["width"], state["height"]
            current = effective_tiles(load_tiles(conn))
            fingerprint = compute_fingerprint(width, height, current)
        path = preview_dir_for(fingerprint) / PNG_NAME
        if not path.is_file():
            raise HTTPException(status_code=404, detail="预览图不存在。")
        return FileResponse(
            path,
            media_type="image/png",
            headers={"Cache-Control": "no-store", "X-Mosaic-Fingerprint": fingerprint},
        )

    @app.get("/preview.csv")
    def preview_csv() -> FileResponse:
        with db_lock as conn:
            state = read_preview_state(conn)
            if state["version"] != state["state_version"]:
                raise HTTPException(status_code=409, detail="预览尚未生成完成。")
            width, height = state["width"], state["height"]
            current = effective_tiles(load_tiles(conn))
            fingerprint = compute_fingerprint(width, height, current)
        path = preview_dir_for(fingerprint) / CSV_NAME
        if not path.is_file():
            raise HTTPException(status_code=404, detail="来源 CSV 不存在。")
        return FileResponse(
            path,
            media_type="text/csv; charset=utf-8",
            filename=f"preview-{fingerprint[:12]}.csv",
            headers={"Cache-Control": "no-store"},
        )

    # ------------------------------------------------------------------ #
    # Published downloads
    # ------------------------------------------------------------------ #

    def _find_version(version: str) -> Any:
        with db_lock as conn:
            if version == "latest":
                row = conn.execute(
                    "SELECT * FROM published_versions ORDER BY seq DESC LIMIT 1"
                ).fetchone()
            elif version.isdigit():
                row = conn.execute(
                    "SELECT * FROM published_versions WHERE seq = ?", (int(version),)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM published_versions WHERE id = ?", (version,)
                ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="未找到该发布版本。")
        return row

    def _deliver(version: str, column: str, media_type: str, filename: str):
        row = _find_version(version)
        relative = row[column]
        path = (runtime.versions_dir / relative).resolve()
        # Versions only become visible after an atomic directory rename, so a
        # recorded version must be complete.  A staging directory is never
        # served even if a stray one is left behind by a crashed process.
        if (
            not path.is_file()
            or path.parent.name.startswith(STAGING_PREFIX)
            or path.parent.parent != runtime.versions_dir.resolve()
        ):
            raise HTTPException(status_code=503, detail="该版本文件不完整。")
        data = path.read_bytes()
        expected = row["png_sha256"] if column == "png_rel" else row["csv_sha256"]
        if hashlib.sha256(data).hexdigest() != expected:
            raise HTTPException(status_code=503, detail="该版本校验失败。")
        from fastapi.responses import Response

        return Response(
            content=data,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(data)),
                "Cache-Control": "no-store",
                "X-Mosaic-Version": f"{row['seq']}",
                "X-Mosaic-Sha256": expected,
            },
        )

    @app.get("/download/{version}/mosaic.png")
    def download_png(version: str):
        return _deliver(version, "png_rel", "image/png", "mosaic.png")

    @app.get("/download/{version}/sources.csv")
    def download_csv(version: str):
        return _deliver(
            version, "csv_rel", "text/csv; charset=utf-8", "sources.csv"
        )

    @app.get("/download/{version}/manifest.json")
    def download_manifest(version: str) -> JSONResponse:
        row = _find_version(version)
        path = runtime.versions_dir / row["dir_rel"] / "manifest.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="未找到清单。")
        return JSONResponse(
            __import__("json").loads(path.read_text(encoding="utf-8")),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/versions")
    def list_versions() -> JSONResponse:
        with db_lock as conn:
            rows = conn.execute(
                "SELECT id, seq, created_at, fingerprint, canvas_width, canvas_height, "
                "tile_count, covered_pixels, png_sha256, csv_sha256 "
                "FROM published_versions ORDER BY seq DESC"
            ).fetchall()
        return JSONResponse([dict(row) for row in rows])

    return app


app = create_app(os.environ.get("MOSAIC_DATA_DIR", "./data"))

"""Mosaic behaviour: cloud masking, revision selection, ordering, provenance."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.mosaic import (
    TileRecord,
    compute_fingerprint,
    effective_tiles,
    render_mosaic,
)
from tests.conftest import fetch_csv, fetch_png, mask_bytes, png_bytes


def _write_tile(storage: Path, name: str, colour, w, h, cloud) -> tuple[str, str]:
    storage.mkdir(parents=True, exist_ok=True)
    image_rel = f"{name}.png"
    mask_rel = f"{name}.mask.png"
    (storage / image_rel).write_bytes(png_bytes(w, h, colour))
    (storage / mask_rel).write_bytes(mask_bytes(w, h, cloud))
    return image_rel, mask_rel


def _record(rid, tile_id, revision, x, y, w, h, image_rel, mask_rel, submitted_at):
    return TileRecord(
        id=rid,
        tile_id=tile_id,
        revision=revision,
        x=x,
        y=y,
        width=w,
        height=h,
        image_rel=image_rel,
        mask_rel=mask_rel,
        image_sha256="",
        mask_sha256="",
        cloud_pixels=0,
        clear_pixels=w * h,
        submitted_at=submitted_at,
    )


def test_cloud_pixels_are_transparent_and_never_a_source(tmp_path):
    storage = tmp_path / "tiles"
    image_rel, mask_rel = _write_tile(storage, "a", (255, 0, 0), 3, 3, {(0, 0), (1, 1)})
    tile = _record(1, "A", 1, 0, 0, 3, 3, image_rel, mask_rel, "2026-01-01T00:00:00+00:00")

    png_path = tmp_path / "out.png"
    csv_path = tmp_path / "out.csv"
    stats = render_mosaic(3, 3, [tile], storage, png_path, csv_path)

    image = Image.open(png_path).convert("RGBA")
    assert image.getpixel((0, 0))[3] == 0
    assert image.getpixel((1, 1))[3] == 0
    assert image.getpixel((2, 2))[:3] == (255, 0, 0)
    assert image.getpixel((2, 2))[3] == 255
    assert stats.covered_pixels == 7
    assert stats.transparent_pixels == 2

    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "x,y,tile_id,revision"
    assert "0,0,," in rows  # cloud pixel has no source
    assert "2,2,A,1" in rows


def test_newest_submission_wins_and_cloudy_tile_falls_back(tmp_path):
    storage = tmp_path / "tiles"
    # Older tile is fully clear.
    old_rel = _write_tile(storage, "old", (0, 0, 255), 2, 2, set())
    # Newer tile covers the same area but its left column is cloudy.
    new_rel = _write_tile(storage, "new", (0, 255, 0), 2, 2, {(0, 0), (0, 1)})
    old = _record(1, "OLD", 1, 0, 0, 2, 2, *old_rel, "2026-01-01T00:00:00+00:00")
    new = _record(2, "NEW", 1, 0, 0, 2, 2, *new_rel, "2026-01-02T00:00:00+00:00")

    png_path = tmp_path / "out.png"
    csv_path = tmp_path / "out.csv"
    render_mosaic(2, 2, [old, new], storage, png_path, csv_path)

    image = Image.open(png_path).convert("RGBA")
    # Newer clear pixel wins.
    assert image.getpixel((1, 0))[:3] == (0, 255, 0)
    # Newer cloudy pixel: older clear tile backfills it.
    assert image.getpixel((0, 0))[:3] == (0, 0, 255)
    assert image.getpixel((0, 1))[:3] == (0, 0, 255)

    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert "1,0,NEW,1" in rows
    assert "0,0,OLD,1" in rows


def test_all_sources_cloudy_leaves_pixel_transparent(tmp_path):
    storage = tmp_path / "tiles"
    a_rel = _write_tile(storage, "a", (1, 2, 3), 2, 1, {(0, 0)})
    b_rel = _write_tile(storage, "b", (4, 5, 6), 2, 1, {(0, 0)})
    a = _record(1, "A", 1, 0, 0, 2, 1, *a_rel, "2026-01-01T00:00:00+00:00")
    b = _record(2, "B", 1, 0, 0, 2, 1, *b_rel, "2026-01-02T00:00:00+00:00")
    png_path = tmp_path / "out.png"
    csv_path = tmp_path / "out.csv"
    render_mosaic(2, 1, [a, b], storage, png_path, csv_path)
    image = Image.open(png_path).convert("RGBA")
    assert image.getpixel((0, 0))[3] == 0
    assert image.getpixel((1, 0))[:3] == (4, 5, 6)


def test_only_newest_revision_per_tile_is_effective():
    def r(rid, revision, submitted):
        return _record(rid, "A", revision, 0, 0, 1, 1, "x", "y", submitted)

    records = [
        r(1, 1, "2026-01-03T00:00:00+00:00"),
        r(2, 2, "2026-01-01T00:00:00+00:00"),
    ]
    chosen = effective_tiles(records)
    assert [t.revision for t in chosen] == [2]
    # Withdrawing the newest revision falls back to the previous one.
    withdrawn = [
        records[0],
        TileRecord(**{**records[1].__dict__, "status": "withdrawn"}),
    ]
    assert effective_tiles(withdrawn)[0].revision == 1
    # Withdrawing every revision leaves no effective tile.
    assert effective_tiles(
        [TileRecord(**{**tw.__dict__, "status": "withdrawn"}) for tw in withdrawn]
    ) == []


def test_fingerprint_tracks_masks_and_canvas():
    a = _record(1, "A", 1, 0, 0, 2, 2, "x", "y", "2026-01-01T00:00:00+00:00")
    base = compute_fingerprint(4, 4, [a])
    assert base == compute_fingerprint(4, 4, [a])
    assert base != compute_fingerprint(5, 4, [a])
    changed = TileRecord(**{**a.__dict__, "mask_sha256": "deadbeef"})
    assert base != compute_fingerprint(4, 4, [changed])


def test_partially_offscreen_tile_is_clipped(tmp_path):
    storage = tmp_path / "tiles"
    rel = _write_tile(storage, "a", (9, 9, 9), 4, 4, set())
    tile = _record(1, "A", 1, -2, -2, 4, 4, *rel, "2026-01-01T00:00:00+00:00")
    png_path = tmp_path / "out.png"
    csv_path = tmp_path / "out.csv"
    stats = render_mosaic(2, 2, [tile], storage, png_path, csv_path)
    assert stats.covered_pixels == 4
    assert Image.open(png_path).convert("RGBA").getpixel((0, 0))[:3] == (9, 9, 9)


# --------------------------------------------------------------------------- #
# End-to-end through the web app
# --------------------------------------------------------------------------- #


def test_web_upload_preview_and_provenance(client, upload, canvas):
    assert canvas(4, 3).status_code == 303
    assert upload(tile_id="A", revision=1, x=0, y=0, colour=(10, 20, 30),
                  width=4, height=3, cloud={(3, 2)}).status_code == 303

    png = fetch_png(client)
    assert png.size == (4, 3)
    rgba = png.convert("RGBA")
    assert rgba.getpixel((0, 0))[:3] == (10, 20, 30)
    assert rgba.getpixel((3, 2))[3] == 0

    rows = fetch_csv(client)
    assert rows[0] == ["x", "y", "tile_id", "revision"]
    assert len(rows) == 1 + 12
    assert rows[1] == ["0", "0", "A", "1"]
    assert ["3", "2", "", ""] in rows


def test_web_higher_revision_replaces_older(client, upload, canvas):
    canvas(2, 2)
    upload(tile_id="T", revision=1, colour=(255, 0, 0), width=2, height=2)
    png = fetch_png(client).convert("RGBA")
    assert png.getpixel((0, 0))[:3] == (255, 0, 0)

    upload(tile_id="T", revision=2, colour=(0, 255, 0), width=2, height=2)
    png = fetch_png(client).convert("RGBA")
    assert png.getpixel((0, 0))[:3] == (0, 255, 0)
    rows = fetch_csv(client)
    assert {row[2] for row in rows[1:]} == {"T"}
    assert {row[3] for row in rows[1:]} == {"2"}


def test_web_withdraw_backfills_from_other_tiles(client, upload, canvas):
    canvas(2, 2)
    upload(tile_id="BASE", revision=1, colour=(0, 0, 255), width=2, height=2)
    upload(tile_id="TOP", revision=1, colour=(0, 200, 0), width=2, height=2)
    assert fetch_png(client).convert("RGBA").getpixel((1, 1))[:3] == (0, 200, 0)

    page = client.get("/")
    import re

    match = re.search(
        r"<td class=\"mono\">TOP</td>(.*?)/tiles/(\d+)/withdraw",
        page.text,
        re.S,
    )
    assert match
    top_id = match.group(2)
    client.post(f"/tiles/{top_id}/withdraw", follow_redirects=False)

    rgba = fetch_png(client).convert("RGBA")
    assert rgba.getpixel((1, 1))[:3] == (0, 0, 255)
    rows = fetch_csv(client)
    assert {row[2] for row in rows[1:]} == {"BASE"}


def test_web_cloud_from_newer_tile_shows_older_clear_pixel(client, upload, canvas):
    canvas(2, 1)
    upload(tile_id="OLD", revision=1, colour=(255, 255, 0), width=2, height=1)
    upload(tile_id="NEW", revision=1, colour=(0, 255, 255), width=2, height=1,
           cloud={(0, 0)})
    rgba = fetch_png(client).convert("RGBA")
    assert rgba.getpixel((0, 0))[:3] == (255, 255, 0)
    assert rgba.getpixel((1, 0))[:3] == (0, 255, 255)


def test_web_rejects_invalid_uploads(client, upload, canvas):
    canvas(5, 5)
    # Mask is the wrong size.
    bad_mask = client.post(
        "/tiles",
        files={
            "image": ("a.png", png_bytes(2, 2), "image/png"),
            "mask": ("m.png", mask_bytes(3, 2), "image/png"),
            "tile_id": (None, "A"),
            "revision": (None, "1"),
            "x": (None, "0"),
            "y": (None, "0"),
        },
        follow_redirects=False,
    )
    assert bad_mask.status_code == 303
    assert "err=" in bad_mask.headers["location"]

    # Duplicate (tile_id, revision).
    assert upload(tile_id="B", revision=1).status_code == 303
    duplicate = upload(tile_id="B", revision=1)
    assert "err=" in duplicate.headers["location"]

    # Non-RGB image.
    grey = client.post(
        "/tiles",
        files={
            "image": ("a.png", png_bytes(2, 2, 128, mode="L"), "image/png"),
            "mask": ("m.png", mask_bytes(2, 2), "image/png"),
            "tile_id": (None, "C"),
            "revision": (None, "1"),
            "x": (None, "0"),
            "y": (None, "0"),
        },
        follow_redirects=False,
    )
    assert "err=" in grey.headers["location"]

    # Still only B in the preview (pixels outside B stay transparent).
    rows = fetch_csv(client)
    assert {row[2] for row in rows[1:] if row[3]} == {"B"}


def test_web_withdraw_latest_revision_falls_back_to_previous(client, upload, canvas):
    canvas(2, 2)
    upload(tile_id="T", revision=1, colour=(255, 0, 0), width=2, height=2)
    upload(tile_id="T", revision=2, colour=(0, 0, 255), width=2, height=2)
    assert fetch_png(client).convert("RGBA").getpixel((0, 0))[:3] == (0, 0, 255)

    page = client.get("/").text
    import re

    r2_only = re.search(
        r"<td class=\"mono\">T</td>\s*<td class=\"num\">r2</td>.*?/tiles/(\d+)/withdraw",
        page,
        re.S,
    )
    assert r2_only
    client.post(f"/tiles/{r2_only.group(1)}/withdraw", follow_redirects=False)
    rgba = fetch_png(client).convert("RGBA")
    assert rgba.getpixel((0, 0))[:3] == (255, 0, 0)
    rows = fetch_csv(client)
    assert {row[3] for row in rows[1:]} == {"1"}


def test_web_cannot_resubmit_withdrawn_revision(client, upload, canvas):
    canvas(1, 1)
    upload(tile_id="T", revision=1, colour=(1, 2, 3), width=1, height=1)
    page = client.get("/").text
    import re

    rid = re.search(r"/tiles/(\d+)/withdraw", page).group(1)
    client.post(f"/tiles/{rid}/withdraw", follow_redirects=False)
    again = upload(tile_id="T", revision=1, colour=(4, 5, 6), width=1, height=1)
    assert "err=" in again.headers["location"]
    # A higher revision is accepted again.
    ok = upload(tile_id="T", revision=2, colour=(4, 5, 6), width=1, height=1)
    assert "msg=" in ok.headers["location"]
    assert fetch_png(client).convert("RGBA").getpixel((0, 0))[:3] == (4, 5, 6)


def test_web_canvas_resize_reclips_preview(client, upload, canvas):
    canvas(4, 2)
    upload(tile_id="T", revision=1, colour=(7, 8, 9), width=4, height=4)
    assert fetch_png(client).size == (4, 2)
    canvas(2, 4)
    image = fetch_png(client)
    assert image.size == (2, 4)
    rows = fetch_csv(client)
    assert len(rows) == 1 + 8


def test_unknown_version_returns_404(client):
    assert client.get("/download/latest/mosaic.png").status_code == 404
    assert client.get("/download/999/mosaic.png").status_code == 404
    assert client.get("/download/nope/sources.csv").status_code == 404

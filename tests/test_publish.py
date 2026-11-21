"""Publishing atomicity: a publish always exposes one complete input set."""

from __future__ import annotations

import hashlib
import io
import threading

from PIL import Image

from app import main as main_module
from tests.conftest import fetch_png, mask_bytes, png_bytes


def _files(tile_id, revision, colour, cloud=()):
    return {
        "image": ("t.png", png_bytes(2, 2, colour), "image/png"),
        "mask": ("m.png", mask_bytes(2, 2, cloud), "image/png"),
        "tile_id": (None, tile_id),
        "revision": (None, str(revision)),
        "x": (None, "0"),
        "y": (None, "0"),
    }


def _publish_and_download(client):
    assert client.post("/publish", follow_redirects=False).status_code == 303
    versions = client.get("/versions").json()
    assert versions
    # Latest published version by sequence number.
    latest = versions[0]
    png = client.get(f"/download/{latest['id']}/mosaic.png")
    csv = client.get(f"/download/{latest['id']}/sources.csv")
    assert png.status_code == 200
    assert csv.status_code == 200
    return latest, png, csv


def test_publish_exports_matching_png_and_csv(client, upload, canvas):
    canvas(2, 2)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2)
    latest, png, csv = _publish_and_download(client)
    assert png.headers["content-type"].startswith("image/png")
    image = Image.open(io.BytesIO(png.content)).convert("RGBA")
    assert image.getpixel((0, 0))[:3] == (255, 0, 0)
    text = csv.content.decode("utf-8")
    assert text.splitlines()[1] == "0,0,A,1"

    versions = client.get("/versions").json()
    assert versions[0]["png_sha256"] == hashlib.sha256(png.content).hexdigest()
    assert versions[0]["csv_sha256"] == hashlib.sha256(csv.content).hexdigest()


def test_published_version_is_frozen_after_uploads(client, upload, canvas):
    canvas(2, 2)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2)
    first, png1, csv1 = _publish_and_download(client)

    # Change the draft after publishing.
    upload(tile_id="A", revision=2, colour=(0, 0, 255), width=2, height=2)
    upload(tile_id="B", revision=1, colour=(0, 255, 0), width=2, height=2)

    # The old version is byte-for-byte unchanged.
    again_png = client.get(f"/download/{first['id']}/mosaic.png")
    again_csv = client.get(f"/download/{first['id']}/sources.csv")
    assert again_png.content == png1.content
    assert again_csv.content == csv1.content

    # The preview reflects the newer draft.
    preview = fetch_png(client).convert("RGBA")
    assert preview.getpixel((0, 0))[:3] == (0, 255, 0)


def test_publish_during_concurrent_edit_never_mixes_inputs(client, upload, canvas, monkeypatch):
    canvas(2, 2)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2)

    real_render = main_module.render_mosaic
    fired = threading.Event()

    def slow_render(*args, **kwargs):
        # Mutate the draft while the publish render is in flight.
        fired.set()
        client.post(
            "/tiles",
            files=_files("B", 1, (0, 255, 0)),
            follow_redirects=False,
        )
        return real_render(*args, **kwargs)

    monkeypatch.setattr(main_module, "render_mosaic", slow_render)
    assert client.post("/publish", follow_redirects=False).status_code == 303
    assert fired.is_set()

    latest = client.get("/versions").json()[0]
    png = client.get(f"/download/{latest['id']}/mosaic.png")
    csv = client.get(f"/download/{latest['id']}/sources.csv")
    assert png.status_code == 200 and csv.status_code == 200
    image = Image.open(io.BytesIO(png.content)).convert("RGBA")
    # Snapshot was taken before B existed, so A's red is still the source.
    assert image.getpixel((0, 0))[:3] == (255, 0, 0)
    assert "B," not in csv.content.decode("utf-8")
    assert latest["tile_count"] == 1


def test_failed_publish_keeps_previous_complete_version(client, upload, canvas, monkeypatch):
    canvas(2, 2)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2)
    first, png1, csv1 = _publish_and_download(client)

    upload(tile_id="A", revision=2, colour=(0, 0, 255), width=2, height=2)

    real_render = main_module.render_mosaic

    def boom(*args, **kwargs):
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr(main_module, "render_mosaic", boom)
    response = client.post("/publish", follow_redirects=False)
    assert response.status_code == 303
    assert "err=" in response.headers["location"]

    # No half-finished version was recorded; the download endpoint still
    # serves the last complete one.
    versions = client.get("/versions").json()
    assert len(versions) == 1
    assert versions[0]["id"] == first["id"]
    assert client.get("/download/latest/mosaic.png").content == png1.content
    assert client.get("/download/latest/sources.csv").content == csv1.content

    # And the system recovers once rendering works again.
    monkeypatch.setattr(main_module, "render_mosaic", real_render)
    latest, png2, csv2 = _publish_and_download(client)
    image = Image.open(__import__("io").BytesIO(png2.content)).convert("RGBA")
    assert image.getpixel((0, 0))[:3] == (0, 0, 255)


def test_publish_with_failing_preview_recovers(client, upload, canvas, monkeypatch):
    canvas(2, 2)
    real_render = main_module.render_mosaic

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(main_module, "render_mosaic", boom)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2)
    assert client.get("/preview.png").status_code in (404, 409)
    # Publish must not invent a version when it cannot render.
    client.post("/publish", follow_redirects=False)
    assert client.get("/versions").json() == []

    monkeypatch.setattr(main_module, "render_mosaic", real_render)
    latest, png, csv = _publish_and_download(client)
    assert latest["tile_count"] == 1


def test_published_files_are_immutable_and_content_addressed(client, upload, canvas):
    canvas(1, 1)
    upload(tile_id="A", revision=1, colour=(1, 1, 1), width=1, height=1)
    latest, png, csv = _publish_and_download(client)
    data_dir = client.app.state.runtime.data_dir
    version_files = list((data_dir / "published" / latest["id"]).parent.glob("*"))
    assert any(p.name.endswith(latest["id"]) or latest["id"] in p.name
               for p in version_files)
    manifest = client.get(f"/download/{latest['id']}/manifest.json").json()
    assert manifest["fingerprint"] == latest["fingerprint"]
    assert manifest["tiles"][0]["tile_id"] == "A"

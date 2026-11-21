"""Test helpers: build in-memory PNG uploads and a running app client."""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Iterable

import pytest
from PIL import Image
from starlette.testclient import TestClient

from app.main import create_app

warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")


def png_bytes(
    width: int,
    height: int,
    colour=(255, 255, 255),
    *,
    mode: str = "RGB",
) -> bytes:
    """Uniform image; `colour` may be a scalar for single-channel modes."""
    image = Image.new(mode, (width, height), colour)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def mask_bytes(width: int, height: int, cloud: Iterable[tuple[int, int]] = ()) -> bytes:
    """Binary mask, 255 = cloud, 0 = usable; same size as the tile."""
    image = Image.new("L", (width, height), 0)
    pixels = image.load()
    for x, y in cloud:
        pixels[x, y] = 255
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def tile_upload(
    files: dict,
    *,
    tile_id: str,
    revision: int,
    x: int,
    y: int,
    image: bytes,
    mask: bytes,
) -> None:
    files["image"] = ("tile.png", image, "image/png")
    files["mask"] = ("mask.png", mask, "image/png")
    files["tile_id"] = (None, tile_id)
    files["revision"] = (None, str(revision))
    files["x"] = (None, str(x))
    files["y"] = (None, str(y))


@pytest.fixture()
def app_client(tmp_path: Path):
    app = create_app(tmp_path / "data")
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def client(app_client):
    return app_client


@pytest.fixture()
def upload(client):
    def _upload(
        *,
        tile_id: str,
        revision: int = 1,
        x: int = 0,
        y: int = 0,
        colour=(200, 30, 30),
        width: int = 4,
        height: int = 4,
        cloud: Iterable[tuple[int, int]] = (),
        image: bytes | None = None,
        mask: bytes | None = None,
    ):
        files = {}
        tile_upload(
            files,
            tile_id=tile_id,
            revision=revision,
            x=x,
            y=y,
            image=image if image is not None else png_bytes(width, height, colour),
            mask=(
                mask if mask is not None
                else mask_bytes(width, height, cloud)
            ),
        )
        return client.post("/tiles", files=files, follow_redirects=False)

    return _upload


@pytest.fixture()
def canvas(client):
    def _set(width: int, height: int):
        return client.post(
            "/canvas",
            data={"width": str(width), "height": str(height)},
            follow_redirects=False,
        )

    return _set


def fetch_png(client, url: str = "/preview.png") -> Image.Image:
    response = client.get(url)
    assert response.status_code == 200, response.text
    return Image.open(io.BytesIO(response.content))


def fetch_csv(client, url: str = "/preview.csv") -> list[list[str]]:
    response = client.get(url)
    assert response.status_code == 200, response.text
    text = response.content.decode("utf-8")
    import csv

    return list(csv.reader(io.StringIO(text)))

"""Upload validation and canvas bounds."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.mosaic import (
    TileValidationError,
    validate_canvas,
    validate_revision,
    validate_tile_id,
    validate_upload,
)
from tests.conftest import mask_bytes, png_bytes


def _upload(**overrides):
    kwargs = dict(
        tile_id="T1",
        revision="1",
        x="0",
        y="0",
        image_bytes=png_bytes(2, 2),
        mask_bytes=mask_bytes(2, 2),
    )
    kwargs.update(overrides)
    return validate_upload(**kwargs)


@pytest.mark.parametrize("bad", ["", " ", " bad id", "../x", "a/b", "x" * 65, "ä"])
def test_bad_tile_ids_rejected(bad):
    with pytest.raises(TileValidationError):
        validate_tile_id(bad)


@pytest.mark.parametrize("good", ["A", "N37E127", "a_b.c-1", "9"])
def test_good_tile_ids_accepted(good):
    assert validate_tile_id(good) == good


@pytest.mark.parametrize("bad", ["0", "-1", "1.5", "", "x"])
def test_bad_revisions_rejected(bad):
    with pytest.raises(TileValidationError):
        validate_revision(bad)


def test_canvas_bounds():
    validate_canvas(1, 1)
    validate_canvas(20000, 1)
    for width, height in [(0, 5), (5, 0), (20001, 1), (9000, 9000)]:
        with pytest.raises(TileValidationError):
            validate_canvas(width, height)


def test_mask_must_match_image_size():
    with pytest.raises(TileValidationError):
        _upload(mask_bytes=mask_bytes(3, 2))


def test_mask_must_be_binary_capable_mode():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    with pytest.raises(TileValidationError):
        _upload(mask_bytes=buffer.getvalue())


def test_image_must_be_rgb_png():
    grey = png_bytes(2, 2, 120, mode="L")
    with pytest.raises(TileValidationError):
        _upload(image_bytes=grey)
    # JPEG content with a .png name is rejected by real format detection.
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="JPEG")
    with pytest.raises(TileValidationError):
        _upload(image_bytes=buffer.getvalue())


def test_binary_mask_counts_cloud_pixels():
    result = _upload(
        image_bytes=png_bytes(3, 3),
        mask_bytes=mask_bytes(3, 3, {(0, 0), (2, 2), (1, 1)}),
    )
    assert result.cloud_pixels == 3
    assert result.clear_pixels == 6

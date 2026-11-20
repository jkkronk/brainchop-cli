"""brainchop.niimath against the niimath binary, on synthetic volumes (no model needed)."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from brainchop.niimath import bwlabel, conform, truncate_header_bytes

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "brain.nii.gz"


@pytest.fixture(scope="module")
def header():
    """Header of the conformed sample volume: a 256^3 uint8 grid with 1 mm voxels."""
    return conform(str(EXAMPLE))[1]


def test_bwlabel_without_foreground_returns_input(header, monkeypatch):
    calls = []
    run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or run(*a, **k))
    vol = np.zeros((256, 256, 256), np.uint8)
    out, out_header = bwlabel(header, vol)
    assert calls == []
    assert out.dtype == np.uint8 and out.shape == vol.shape and not out.any()
    assert out_header == truncate_header_bytes(header)


@pytest.mark.parametrize(
    "islands, blob_z, islands_z",
    [(3000, 100, 50), (3000, 50, 100), (200, 50, 100)],
    ids=["over-255-largest-scanned-last", "over-255-largest-scanned-first", "under-255"],
)
def test_bwlabel_keeps_only_the_largest_component(header, islands, blob_z, islands_z):
    vol = np.zeros((256, 256, 256), np.uint8)
    vol[blob_z : blob_z + 10, 100:110, 100:105] = 1  # 1000-voxel component holding labels 1 and 2
    vol[blob_z : blob_z + 10, 100:110, 105:110] = 2
    expected = vol.copy()
    plane = vol[islands_z, ::2, ::2]  # single voxels two apart: one component each
    plane[np.unravel_index(np.arange(islands), plane.shape)] = 1
    out, out_header = bwlabel(header, vol)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, expected)
    assert out_header == truncate_header_bytes(header)

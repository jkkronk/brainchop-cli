"""brainchop.niimath against the niimath binary, on synthetic volumes (no model needed)."""

import gzip
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import brainchop.niimath
from brainchop.niimath import (
    bwlabel,
    conform,
    grow_border,
    header_data_split,
    truncate_header_bytes,
)

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


def _within(mask, n):
    """Voxels within Euclidean distance n of the mask (mask voxels at least n from the edge)."""
    r2 = np.arange(-n, n + 1) ** 2
    ball = r2[:, None, None] + r2[None, :, None] + r2[None, None, :] <= n * n
    out = np.zeros(mask.shape, bool)
    for z, y, x in np.argwhere(mask):
        out[z - n : z + n + 1, y - n : y + n + 1, x - n : x + n + 1] |= ball
    return out


@pytest.mark.parametrize("border", [1, 2, 3])
@pytest.mark.parametrize("seed", ["voxel", "cube"])
def test_grow_border_grows_by_border_mm_inclusive(header, seed, border):
    mask = np.zeros((256, 256, 256), np.uint8)
    if seed == "voxel":
        mask[128, 128, 128] = 1
    else:
        mask[100:104, 120:124, 90:94] = 1
    full_input = truncate_header_bytes(header) + mask.tobytes()
    out_header, data = header_data_split(grow_border(full_input, border))
    grown = np.frombuffer(data, np.uint8).reshape(mask.shape)
    assert out_header == truncate_header_bytes(header)
    assert set(np.unique(grown)) == {0, 1}
    np.testing.assert_array_equal(grown.astype(bool), _within(mask, border))


CHILD = (
    "import json, shutil, sys\n"
    "before = shutil.which('niimath')\n"
    "import brainchop.niimath\n"
    "after = shutil.which('niimath')\n"
    "print(json.dumps([sys.executable, brainchop.niimath.__file__, before, after]))\n"
)


def _niimath_in_child(path, niimath_path=None):
    """Where a fresh interpreter finds niimath before and after importing brainchop.niimath."""
    env = dict(os.environ, PATH=path)
    env.pop("NIIMATH_PATH", None)
    if niimath_path is not None:
        env["NIIMATH_PATH"] = niimath_path
    res = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    exe, module, before, after = json.loads(res.stdout)
    assert Path(module).resolve() == Path(brainchop.niimath.__file__).resolve()  # code under test
    return exe, before, after


def _fake_niimath(directory):
    directory.mkdir()
    exe = directory / "niimath"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    return str(exe)


@pytest.mark.usefixtures("home")
@pytest.mark.parametrize("bogus_niimath_path", [False, True])
def test_import_finds_niimath_next_to_the_interpreter(tmp_path, bogus_niimath_path):
    # pipx / uv tool / cron: the interpreter's directory, where pip put niimath, is not on PATH
    empty = tmp_path / "empty"
    empty.mkdir()
    exe, before, after = _niimath_in_child(str(empty), str(empty) if bogus_niimath_path else None)
    python_dir = os.path.dirname(exe)
    if not shutil.which("niimath", path=python_dir):
        pytest.skip(f"no niimath launcher next to {exe}")
    assert before is None
    assert after == os.path.join(python_dir, "niimath")


@pytest.mark.usefixtures("home")
def test_niimath_path_goes_first_on_path(tmp_path):
    on_path = _fake_niimath(tmp_path / "on_path")
    override = _fake_niimath(tmp_path / "override")
    _, before, after = _niimath_in_child(os.path.dirname(on_path), os.path.dirname(override))
    assert (before, after) == (on_path, override)


@pytest.mark.usefixtures("home")
def test_niimath_path_without_niimath_is_ignored(tmp_path):
    on_path = _fake_niimath(tmp_path / "on_path")
    _, before, after = _niimath_in_child(os.path.dirname(on_path), str(tmp_path / "missing"))
    assert before == after == on_path


def test_comply_rescales_wide_range_input(brain_path, tmp_path):
    src_header, data = header_data_split(gzip.decompress(brain_path.read_bytes()))
    int16_header = bytearray(src_header)
    struct.pack_into("<hh", int16_header, 70, 4, 16)  # datatype int16, bitpix 16
    voxels = np.frombuffer(data, np.uint8).astype(np.int16) * 8
    x8 = tmp_path / "brain_x8_int16.nii"
    x8.write_bytes(bytes(int16_header) + voxels.tobytes())
    vol, _ = conform(str(x8), comply=True)
    assert (vol[vol > 0] == 255).mean() < 0.1
    assert np.unique(vol).size > 100
    np.testing.assert_array_equal(vol, conform(str(brain_path), comply=True)[0])

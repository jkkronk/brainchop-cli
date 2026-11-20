"""CLI behavior with the stub model: output naming, batching and the mindgrab outputs.

Each CLI run still conforms, labels and saves real 256^3 volumes, so it costs a few
seconds per input; keep the number of inputs per test small.
"""

import gzip
import shutil
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from brainchop import cli

GZIP_MAGIC = b"\x1f\x8b"
NIFTI_DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64,
                256: np.int8, 512: np.uint16, 768: np.uint32, 1024: np.int64, 1280: np.uint64}


def read_nifti(path):
    """Shape, datatype, scaling and stored voxels (x fastest) of a single-file NIfTI-1."""
    raw = Path(path).read_bytes()
    if raw[:2] == GZIP_MAGIC:
        raw = gzip.decompress(raw)
    dim = struct.unpack("<8h", raw[40:56])
    datatype = struct.unpack("<h", raw[70:72])[0]
    vox_offset, slope, inter = struct.unpack("<3f", raw[108:120])
    shape = dim[1:dim[0] + 1]
    data = np.frombuffer(raw, NIFTI_DTYPES[datatype], count=int(np.prod(shape)),
                         offset=int(vox_offset))
    return SimpleNamespace(header=raw[:352], shape=shape, datatype=datatype,
                           slope=slope, inter=inter, data=data)


def copies(src, directory, *names):
    """Copy src to directory/<name>.nii.gz for each name and return the new paths."""
    directory.mkdir(exist_ok=True)
    paths = [directory / f"{name}.nii.gz" for name in names]
    for path in paths:
        shutil.copy(src, path)
    return paths


def listing(directory):
    return sorted(p.name for p in directory.iterdir())


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["brainchop", *map(str, args), "--no-optimize"])
    cli.main()


@pytest.fixture
def workdir(tmp_path, monkeypatch, stub_model):
    """An empty current directory for the CLI's outputs, with the stub model in place."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


@pytest.mark.parametrize("output, gzipped", [("output.nii.gz", True), ("custom.nii", False)])
def test_output_flag_names_the_single_output(workdir, brain_path, monkeypatch, output, gzipped):
    run_cli(monkeypatch, brain_path, "-o", output)
    assert listing(workdir) == [output]
    assert ((workdir / output).read_bytes()[:2] == GZIP_MAGIC) == gzipped
    assert read_nifti(workdir / output).shape == (256, 256, 256)


def test_without_output_flag_the_output_is_auto_named(workdir, brain_path, monkeypatch):
    run_cli(monkeypatch, brain_path)
    assert listing(workdir) == ["brain_tissue_fast_1.nii.gz"]


def test_output_flag_with_several_inputs_keeps_auto_names_and_says_so(
    workdir, tmp_path, brain_path, monkeypatch, capsys
):
    inputs = copies(brain_path, tmp_path / "in", "a", "b")
    run_cli(monkeypatch, *inputs, "-o", "x.nii.gz")
    assert listing(workdir) == ["a_tissue_fast_1.nii.gz", "b_tissue_fast_2.nii.gz"]
    notices = [line for line in capsys.readouterr().err.splitlines() if "x.nii.gz" in line]
    assert len(notices) == 1

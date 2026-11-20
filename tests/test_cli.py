"""CLI behavior with the stub model: output naming, batching and the mindgrab outputs.

Each CLI run still conforms, labels and saves real 256^3 volumes, so it costs a few
seconds per input; keep the number of inputs per test small.
"""

import gzip
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from brainchop import cli

GZIP_MAGIC = b"\x1f\x8b"
NIFTI_DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64,
                256: np.int8, 512: np.uint16, 768: np.uint32, 1024: np.int64, 1280: np.uint64}
ONE_VOLUME = (1, 1, 256, 256, 256)


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


@pytest.fixture
def half_brain(tmp_path, brain_path):
    """The sample head on a 2 mm grid (128^3): another grid, and another shape under --crop."""
    path = tmp_path / "in" / "half.nii.gz"
    path.parent.mkdir(exist_ok=True)
    subprocess.run(["niimath", brain_path, "-subsamp2", path], check=True, capture_output=True)
    return path


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


@pytest.fixture
def mirrored_brain(tmp_path, brain_path):
    """The sample mirrored left-right: same grid, different labels."""
    image = read_nifti(brain_path)
    path = tmp_path / "in" / "b.nii"
    path.parent.mkdir(exist_ok=True)
    mirrored = image.data.reshape(image.shape[::-1])[:, :, ::-1]
    path.write_bytes(image.header + mirrored.tobytes())  # the header has vox_offset 352
    return path


def test_batch_size_segments_each_group_in_one_model_call(
    workdir, tmp_path, brain_path, mirrored_brain, stub_model, monkeypatch
):
    # Different scans share the first batch, so results written to the wrong input show
    a, c = copies(brain_path, tmp_path / "in", "a", "c")
    inputs = [a, mirrored_brain, c]
    names = ["a_tissue_fast_1.nii.gz", "b_tissue_fast_2.nii.gz", "c_tissue_fast_3.nii.gz"]
    run_cli(monkeypatch, *inputs)
    assert [call["shape"] for call in stub_model.calls] == [ONE_VOLUME] * 3
    assert not np.array_equal(read_nifti(workdir / names[0]).data,
                              read_nifti(workdir / names[1]).data)

    batched = tmp_path / "batched"
    batched.mkdir()
    monkeypatch.chdir(batched)
    stub_model.calls.clear()
    run_cli(monkeypatch, *inputs, "--batch-size", "2")
    assert [call["shape"] for call in stub_model.calls] == [(2, 1, 256, 256, 256), ONE_VOLUME]
    assert listing(batched) == listing(workdir) == names
    for name in names:
        expected = read_nifti(workdir / name).data
        assert expected.any()
        np.testing.assert_array_equal(read_nifti(batched / name).data, expected)


def test_batch_size_one_segments_each_input_alone(workdir, tmp_path, brain_path, stub_model,
                                                  monkeypatch):
    inputs = copies(brain_path, tmp_path / "in", "a", "b", "c")
    run_cli(monkeypatch, *inputs, "--batch-size", "1")
    assert [call["shape"] for call in stub_model.calls] == [ONE_VOLUME] * 3
    assert listing(workdir) == ["a_tissue_fast_1.nii.gz", "b_tissue_fast_2.nii.gz",
                                "c_tissue_fast_3.nii.gz"]


def test_non_positive_batch_size_counts_as_one(workdir, brain_path, stub_model, monkeypatch):
    run_cli(monkeypatch, brain_path, "--batch-size", "0")
    assert [call["shape"] for call in stub_model.calls] == [ONE_VOLUME]
    assert listing(workdir) == ["brain_tissue_fast_1.nii.gz"]


def test_batch_of_differently_cropped_volumes_is_segmented_one_at_a_time(
    workdir, brain_path, half_brain, stub_model, monkeypatch
):
    run_cli(monkeypatch, brain_path, half_brain, "--crop", "--batch-size", "2")
    shapes = [call["shape"] for call in stub_model.calls]
    assert len(shapes) == 2 and shapes[0][0] == shapes[1][0] == 1
    assert shapes[0] != shapes[1]  # otherwise this would not exercise the fallback
    assert listing(workdir) == ["brain_tissue_fast_1.nii.gz", "half_tissue_fast_2.nii.gz"]


@pytest.mark.parametrize("args, outputs", [
    (["-o", "brain.nii", "--mask", "m.nii"], ["brain.nii", "m.nii"]),
    (["--mask", "custom.nii.gz"], ["brain_mindgrab_1.nii.gz", "custom.nii.gz"]),
], ids=["nii", "nii.gz"])
def test_single_input_mask_is_written_as_named(workdir, brain_path, monkeypatch, args, outputs):
    run_cli(monkeypatch, brain_path, "--skull-strip", *args)
    assert listing(workdir) == outputs
    for name in outputs:
        assert ((workdir / name).read_bytes()[:2] == GZIP_MAGIC) == name.endswith(".gz"), name


def test_several_inputs_get_one_mask_each_on_their_own_grid(
    workdir, brain_path, half_brain, monkeypatch, capsys
):
    run_cli(monkeypatch, brain_path, half_brain, "--skull-strip", "--mask")
    masks = {brain_path: "brain_mindgrab_1_mask.nii.gz", half_brain: "half_mindgrab_2_mask.nii.gz"}
    assert listing(workdir) == sorted(["brain_mindgrab_1.nii.gz", "half_mindgrab_2.nii.gz",
                                       *masks.values()])
    for source, mask in masks.items():
        assert read_nifti(workdir / mask).shape == read_nifti(source).shape
    notices = [line for line in capsys.readouterr().err.splitlines() if "--mask" in line]
    assert len(notices) == 1

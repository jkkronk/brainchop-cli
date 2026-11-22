"""CLI behavior with the stub model: output naming, batching and the mindgrab outputs.

Each CLI run still conforms, labels and saves real 256^3 volumes, so it costs a few
seconds per input; keep the number of inputs per test small.
"""

import functools
import gzip
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.helpers import BEAM

from brainchop import api, cli
from brainchop.api import Volume, load

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


def write_nifti(path, header, data, slope=1.0, inter=0.0):
    """An uncompressed NIfTI-1 with header's geometry, storing data in its own datatype."""
    h = bytearray(header[:352])
    code = next(c for c, t in NIFTI_DTYPES.items() if np.dtype(t) == data.dtype)
    struct.pack_into("<hh", h, 70, code, data.dtype.itemsize * 8)
    struct.pack_into("<3f", h, 108, 352.0, slope, inter)
    h[348:352] = bytes(4)
    Path(path).write_bytes(bytes(h) + data.tobytes())


def real_values(image):
    """Voxel values with the header's scaling applied (none when scl_slope is 0)."""
    if image.slope == 0:
        return image.data.astype(np.float64)
    return image.data * np.float64(image.slope) + image.inter


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


@functools.cache
def mindgrab_stand_in(path):
    """The sample conformed and thresholded, standing in for mindgrab's result.

    Cached, since conforming takes about a second.
    """
    vol = load(path)
    return Volume((vol.data > 60).cast("uint8"), vol.header)


@pytest.fixture
def skull_strip(tmp_path, home, brain_path):
    """A function that runs _save_mindgrab on a small (64^3) source made from the sample.

    The source stores the sample's values times `scale` as `dtype`, with scl_slope `slope`.
    The function returns the source and the skull-stripped image as read back, and the
    mask on the source grid.
    """
    result = mindgrab_stand_in(brain_path)
    template = tmp_path / "template.nii"
    subprocess.run(["niimath", brain_path, "-subsamp2", "-subsamp2", "-gz", "0", template],
                   check=True, capture_output=True)
    template = read_nifti(template)
    source, brain, mask = (str(tmp_path / name) for name in ("source.nii", "brain.nii", "mask.nii"))

    def run(dtype, scale=1.0, slope=1.0):
        write_nifti(source, template.header, np.round(template.data * scale).astype(dtype), slope)
        cli._save_mindgrab(result, source, SimpleNamespace(border=0), brain, mask)
        return read_nifti(source), read_nifti(brain), read_nifti(mask).data > 0

    return run


@pytest.mark.parametrize("dtype, scale, slope", [(np.int16, 128, 1 / 32767),
                                                 (np.uint16, 16, 0.001)], ids=["int16", "uint16"])
def test_skull_strip_keeps_the_real_values_of_a_scaled_source(skull_strip, dtype, scale, slope):
    source, brain, inside = skull_strip(dtype, scale, slope)
    assert inside.sum() > 10000
    np.testing.assert_allclose(real_values(brain)[inside], real_values(source)[inside], rtol=1e-6)
    assert not brain.data[~inside].any()
    assert brain.datatype == 16


@pytest.mark.parametrize("dtype", [np.int8, np.uint32])
def test_skull_strip_of_a_datatype_niimath_cannot_write_is_saved_as_float(skull_strip, dtype):
    source, brain, inside = skull_strip(dtype, scale=0.5)
    assert brain.datatype == 16 and inside.sum() > 10000
    np.testing.assert_array_equal(brain.data[inside], source.data[inside])


def test_skull_strip_of_an_unscaled_source_keeps_its_datatype(skull_strip):
    source, brain, inside = skull_strip(np.uint8)
    assert brain.datatype == 2 and inside.sum() > 10000
    np.testing.assert_array_equal(brain.data[inside], source.data[inside])
    assert not brain.data[~inside].any()


@pytest.mark.parametrize("name, compress", [("plain.nii.gz", False), ("UPPER.NII.GZ", True)])
def test_skull_strip_reads_the_source_header_by_content(tmp_path, brain_path, name, compress):
    # niimath reads both; going by the name crashed (plain) or picked float (upper case)
    raw = gzip.decompress(Path(brain_path).read_bytes())
    path = tmp_path / name
    path.write_bytes(gzip.compress(raw, compresslevel=1) if compress else raw)
    assert cli._brain_odt(str(path)) == "input_force"


def run_cli_with_prompt(monkeypatch, *args):
    """Like run_cli, but without --no-optimize, so the first-run prompt can appear."""
    monkeypatch.setattr(sys, "argv", ["brainchop", *map(str, args)])
    cli.main()


@pytest.fixture
def answer_yes(monkeypatch):
    """Answer the first-run optimization prompt with "y"; returns the prompts shown."""
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


@pytest.fixture
def no_beam_search(monkeypatch, stub_model):
    """Run the real optimize() and segment() without starting a BEAM kernel search.

    optimize() gets a model whose output is already computed, so its BEAM pass compiles
    nothing, and segment() runs with beam 0. Returns the beam the CLI passed to segment().
    """
    computed = Tensor.zeros(1).contiguous().realize()
    real_optimize, real_segment = api.optimize, api.segment

    def optimize(model, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(api, "_load_model", lambda *a, **k: lambda *a, **k: computed)
            real_optimize(model, **kwargs)

    beams = []

    def segment(*args, beam=0, **kwargs):
        beams.append(beam)
        return real_segment(*args, beam=0, **kwargs)

    monkeypatch.setattr(cli, "optimize", optimize)
    monkeypatch.setattr(cli, "segment", segment)
    return beams


def test_custom_model_is_offered_optimization_only_on_its_first_run(
    workdir, brain_path, monkeypatch, answer_yes, no_beam_search
):
    # optimize() caches under the folder's name; the CLI used to look under file://<path>
    (workdir / "mymodel").mkdir()
    for _ in range(2):
        run_cli_with_prompt(monkeypatch, brain_path, "--custom", "mymodel", "-o", "seg.nii.gz")
    assert len(answer_yes) == 1
    assert no_beam_search == [2, 2]


def test_non_positive_batch_size_shares_the_batch_size_one_optimization(
    workdir, brain_path, monkeypatch, answer_yes, no_beam_search
):
    run_cli_with_prompt(monkeypatch, brain_path, "-o", "a.nii.gz")
    run_cli_with_prompt(monkeypatch, brain_path, "-o", "b.nii.gz", "--batch-size", "0")
    assert len(answer_yes) == 1
    assert no_beam_search == [2, 2]


def test_directory_named_like_the_model_does_not_rename_the_output(
    workdir, brain_path, monkeypatch
):
    # People name output folders after models; the output is still labelled by the name
    (workdir / "tissue_fast").mkdir()
    run_cli(monkeypatch, brain_path)
    assert listing(workdir) == ["brain_tissue_fast_1.nii.gz", "tissue_fast"]


class Stop(Exception):
    pass


@pytest.fixture
def beam_probe(home, tmp_path, monkeypatch):
    """Records tinygrad's BEAM level when the model is called, then stops the run.

    Stopping before anything is computed means no kernel is compiled, so a BEAM level
    above 0 cannot start a real kernel search. tissue_fast has a cached BEAM level of 2,
    as after answering "y" to the first-run prompt.
    """
    api._save_optimization_cache("tissue_fast", 1, 2)
    beams = []

    class Model:
        n_classes = 3

        def __call__(self, *args, **kwargs):
            beams.append(BEAM.value)
            raise Stop

    monkeypatch.setattr(api, "_load_model", lambda *a, **k: Model())
    monkeypatch.chdir(tmp_path)
    return beams


@pytest.mark.parametrize("flags", [["--no-optimize"], ["--beam", "0"]])
def test_beam_flags_turn_a_cached_beam_level_off(beam_probe, brain_path, monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["brainchop", str(brain_path), *flags])
    with pytest.raises(Stop):
        cli.main()
    assert beam_probe == [BEAM.value]  # left as tinygrad had it


def test_cached_beam_level_is_used_and_announced(beam_probe, brain_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["brainchop", str(brain_path)])
    with pytest.raises(Stop):
        cli.main()
    assert beam_probe == [2]
    assert "Using BEAM=2" in capsys.readouterr().out

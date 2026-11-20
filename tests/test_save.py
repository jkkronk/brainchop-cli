"""save() to a pathlib.Path, with the suffix choosing gzip or plain NIfTI output."""

import gzip

from brainchop import api

NIFTI1_MAGIC = b"n+1\0"  # bytes 344-347 of a single-file NIfTI-1 header


def test_save_accepts_a_path(brain_path, tmp_path):
    vol = api.load(brain_path)
    api.save(vol, tmp_path / "out.nii.gz")
    api.save(vol, tmp_path / "out.nii")

    compressed = (tmp_path / "out.nii.gz").read_bytes()
    assert compressed[:2] == b"\x1f\x8b"  # gzip magic
    assert gzip.decompress(compressed)[344:348] == NIFTI1_MAGIC
    assert (tmp_path / "out.nii").read_bytes()[344:348] == NIFTI1_MAGIC

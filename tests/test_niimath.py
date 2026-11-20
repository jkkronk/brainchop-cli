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

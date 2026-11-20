"""Shared fixtures for tests that need neither a GPU, a network, nor downloaded weights."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from tinygrad import Tensor

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A private HOME seeded with the repo's models.json.

    brainchop keeps its model listing, weights and optimization caches under
    ~/.cache/brainchop; pointing HOME at a temp dir keeps tests off the network
    and away from the developer's real cache.
    """
    home_dir = tmp_path / "home"
    cache = home_dir / ".cache" / "brainchop"
    cache.mkdir(parents=True)
    shutil.copy(REPO / "models.json", cache / "models.json")
    monkeypatch.setenv("HOME", str(home_dir))
    return home_dir


@pytest.fixture
def brain_path():
    """The sample T1 volume shipped in examples/."""
    return REPO / "examples" / "brain.nii.gz"


class StubModel:
    """Stands in for a loaded network: thresholds its (normalized) input into labels 0/1/2.

    Records the input shape and fuse_chunk of every call, so tests can check how the
    API and CLI drive the model without paying for a 256^3 inference.
    """

    n_classes = 3

    def __init__(self):
        self.calls = []

    def normalize(self, x):
        from brainchop.tiny_meshnet import qnormalize

        return qnormalize(x)

    def __call__(self, x, fuse_chunk=None, **kwargs):
        self.calls.append({"shape": tuple(x.shape), "fuse_chunk": fuse_chunk})
        xn = x.numpy()[:, 0]
        return Tensor((xn > 0.3).astype(np.float32) + (xn > 0.6).astype(np.float32))


@pytest.fixture
def stub_model(home, monkeypatch):
    """Replace model loading with a StubModel (returned so tests can inspect its calls)."""
    import brainchop.api as api

    model = StubModel()
    monkeypatch.setattr(api, "_load_model", lambda *args, **kwargs: model)
    return model

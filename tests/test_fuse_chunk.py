"""FUSE_CHUNK selection: explicit override, cached value, or auto-probing on MemoryError."""

import pytest

from brainchop import api


class ChunkStub:
    """Stand-in model that records each fuse_chunk and runs out of memory above max_chunk."""

    def __init__(self, n_classes, max_chunk=None):
        self.n_classes = n_classes
        self.max_chunk = max_chunk
        self.chunks = []

    def __call__(self, x, fuse_chunk=None):
        self.chunks.append(fuse_chunk)
        if self.max_chunk is not None and fuse_chunk > self.max_chunk:
            raise MemoryError
        return "labels"


def run(model):
    return api._run_with_fuse_chunk(model, None, "tissue_fast")


def cached_chunk():
    return api._load_optimization_cache("tissue_fast").get("fuse_chunk")


@pytest.mark.parametrize("value", [None, "", "0", "-2"])
def test_unset_empty_or_non_positive_means_auto(home, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("FUSE_CHUNK", raising=False)
    else:
        monkeypatch.setenv("FUSE_CHUNK", value)
    model = ChunkStub(n_classes=4)
    assert run(model) == "labels"
    assert model.chunks == [4]
    assert cached_chunk() is None


def test_auto_starts_from_cached_chunk(home, monkeypatch):
    monkeypatch.setenv("FUSE_CHUNK", "0")
    api._save_fuse_chunk("tissue_fast", 2)
    model = ChunkStub(n_classes=4)
    assert run(model) == "labels"
    assert model.chunks == [2]


def test_auto_halves_on_oom_and_caches_what_fits(home, monkeypatch, capsys):
    monkeypatch.delenv("FUSE_CHUNK", raising=False)
    model = ChunkStub(n_classes=7, max_chunk=1)
    assert run(model) == "labels"
    assert model.chunks == [7, 3, 1]
    assert capsys.readouterr().out.splitlines() == [
        "brainchop :: OOM at FUSE_CHUNK=7, retrying with FUSE_CHUNK=3...",
        "brainchop :: OOM at FUSE_CHUNK=3, retrying with FUSE_CHUNK=1...",
        "brainchop :: FUSE_CHUNK=1 (auto-detected for 'tissue_fast', set FUSE_CHUNK to override)",
    ]
    assert cached_chunk() == 1


def test_positive_value_overrides_cache_and_is_never_cached(home, monkeypatch, capsys):
    monkeypatch.setenv("FUSE_CHUNK", "3")
    api._save_fuse_chunk("tissue_fast", 2)
    model = ChunkStub(n_classes=7, max_chunk=1)
    assert run(model) == "labels"
    assert model.chunks == [3, 1]
    assert capsys.readouterr().out.splitlines() == [
        "brainchop :: OOM at FUSE_CHUNK=3, retrying with FUSE_CHUNK=1...",
    ]
    assert cached_chunk() == 2

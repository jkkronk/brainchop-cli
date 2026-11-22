"""tinygrad's BEAM level while segment(), optimize() and export() run a model."""

import sys
from types import ModuleType

import numpy as np
import pytest
import tinygrad.helpers
from tinygrad import Tensor

from brainchop import api


class Stop(Exception):
    pass


class BeamProbe:
    """Stand-in model that records tinygrad's BEAM level each time it is called.

    It then raises Stop (or, with stop=False, returns an already realized tensor), so
    tinygrad never compiles a kernel while BEAM > 0: that would start a real BEAM search.
    """

    n_classes = 3

    def __init__(self):
        self.beams = []
        self.stop = True
        self.output = Tensor.zeros(1).contiguous().realize()

    def __call__(self, x, **kwargs):
        self.beams.append(tinygrad.helpers.BEAM.value)
        if self.stop:
            raise Stop
        return self.output


@pytest.fixture
def probe(home, monkeypatch):
    """Load a BeamProbe as every model, with tinygrad's BEAM preset to 1 as if exported."""
    probe = BeamProbe()
    monkeypatch.setattr(api, "_load_model", lambda *args, **kwargs: probe)
    monkeypatch.setattr(tinygrad.helpers.BEAM, "value", 1)
    return probe


def volume():
    return api.Volume(Tensor(np.zeros((4, 4, 4), np.uint8)), header=b"")


def test_segment_runs_model_at_requested_beam_and_restores_it(probe):
    with pytest.raises(Stop):
        api.segment(volume(), "tissue_fast", beam=2)
    assert probe.beams == [2]
    assert tinygrad.helpers.BEAM.value == 1


def test_segment_without_beam_leaves_tinygrad_beam_alone(probe):
    with pytest.raises(Stop):
        api.segment(volume(), "tissue_fast")  # default beam and nothing cached
    assert probe.beams == [1]
    assert tinygrad.helpers.BEAM.value == 1


def test_segment_uses_the_cached_beam_by_default(probe):
    api._save_optimization_cache("tissue_fast", 1, 2)
    with pytest.raises(Stop):
        api.segment(volume(), "tissue_fast")
    assert probe.beams == [2]
    assert tinygrad.helpers.BEAM.value == 1


def test_segment_with_beam_zero_ignores_the_cached_beam(probe):
    api._save_optimization_cache("tissue_fast", 1, 2)
    with pytest.raises(Stop):
        api.segment(volume(), "tissue_fast", beam=0)
    assert probe.beams == [1]


def test_optimize_runs_model_at_requested_beam_and_restores_it(probe):
    probe.stop = False
    api.optimize("tissue_fast", beam=2)
    assert probe.beams == [2]
    assert tinygrad.helpers.BEAM.value == 1


def test_optimize_restores_beam_when_model_raises(probe):
    with pytest.raises(Stop):
        api.optimize("tissue_fast", beam=2)
    assert probe.beams == [2]
    assert tinygrad.helpers.BEAM.value == 1


def test_export_runs_model_at_requested_beam_and_restores_it(probe, monkeypatch, tmp_path):
    # Stands in for brainchop.export_model, which needs a real export backend.
    exporter = ModuleType("brainchop.export_model")
    exporter.export_model = lambda model, target, *inputs, **kwargs: model(*inputs)
    monkeypatch.setitem(sys.modules, "brainchop.export_model", exporter)
    with pytest.raises(Stop):
        api.export("tissue_fast", str(tmp_path / "export"), beam=2)
    assert probe.beams == [2]
    assert tinygrad.helpers.BEAM.value == 1

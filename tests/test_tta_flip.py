"""TTAModel's matrix-multiply flips on volumes of any size, e.g. loaded with crop=."""

from types import SimpleNamespace

import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict

from brainchop import api

AXES = {"sagittal": 2, "coronal": 3, "axial": 4}  # flip_axis -> axis of (B, C, D, H, W)


def tta_model(flip_axis="sagittal"):
    return api.TTAModel(SimpleNamespace(n_classes=3), flip_axis=flip_axis)


@pytest.mark.parametrize("flip_axis", AXES)
def test_flip_matches_numpy_on_non_cubic_volume(flip_axis):
    x = np.arange(2 * 6 * 5 * 4, dtype=np.float32).reshape(1, 2, 6, 5, 4)
    flipped = tta_model(flip_axis)._flip(Tensor(x)).numpy()
    np.testing.assert_array_equal(flipped, np.flip(x, AXES[flip_axis]))


def test_flip_matrices_are_built_once_per_axis_length(monkeypatch):
    built = []
    create = api._create_flip_matrix

    def counting_create(size):
        built.append(size)
        return create(size)

    monkeypatch.setattr(api, "_create_flip_matrix", counting_create)
    tta = tta_model("axial")
    x = np.arange(2 * 256, dtype=np.float32).reshape(1, 1, 1, 2, 256)
    for _ in range(2):
        np.testing.assert_array_equal(tta._flip(Tensor(x)).numpy(), np.flip(x, 4))
        tta._flip(Tensor.zeros(1, 1, 1, 2, 6))
    assert built == [256, 6]  # the 256 one is pre-created by __init__ and used for 256


def test_exported_state_holds_only_the_precreated_matrix():
    # export_model() saves get_state_dict(model) as the exported weights.
    assert list(get_state_dict(tta_model())) == ["_flip_matrix"]

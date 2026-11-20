"""FP16 unset, empty or "0" keeps weights in float32; any other value loads them as float16."""
import json

import pytest
from tinygrad import Tensor, dtypes, nn
from tinygrad.nn.state import safe_save

from brainchop.sae_model import LAYER_INDICES, SAENet
from brainchop.tiny_meshnet import load_meshnet

LAYER = {"kernel_size": 1, "padding": 0, "stride": 1, "dilation": 1}


@pytest.fixture(params=[(None, dtypes.float32), ("", dtypes.float32), ("0", dtypes.float32),
                        ("1", dtypes.float16), ("true", dtypes.float16)],
                ids=["unset", "empty", "FP16=0", "FP16=1", "FP16=true"])
def weight_dtype(request, monkeypatch):
    """Sets FP16 (None: unset) and returns the dtype the loaded weights must have."""
    value, dtype = request.param
    if value is None:
        monkeypatch.delenv("FP16", raising=False)
    else:
        monkeypatch.setenv("FP16", value)
    return dtype


def test_meshnet_weights(tmp_path, weight_dtype):
    config = {"bnorm": False, "dropout_p": 0, "layers": [
        {"in_channels": 1, "out_channels": 1, **LAYER},
        {"in_channels": 1, "out_channels": 2, **LAYER},
    ]}
    (tmp_path / "model.json").write_text(json.dumps(config))
    safe_save({"model.0.weight": Tensor.ones(1, 1, 1, 1, 1),
               "model.2.weight": Tensor.ones(2, 1, 1, 1, 1)}, str(tmp_path / "model.pth"))
    model = load_meshnet(str(tmp_path / "model.json"), str(tmp_path / "model.pth"))
    assert {t.dtype for t in nn.state.get_state_dict(model.model).values()} == {weight_dtype}


def test_sae_weights(weight_dtype):
    state = {}
    for idx in LAYER_INDICES:
        state[f"{idx}.weight"], state[f"{idx}.bias"] = Tensor.ones(1, 1, 1, 1, 1), Tensor.zeros(1)
    model = SAENet(state, n_classes=1)
    assert {t.dtype for _, _, w, b in model.layers for t in (w, b)} == {weight_dtype}

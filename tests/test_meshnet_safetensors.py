"""Safetensors checkpoints must not load with tensors the model config does not declare."""
import json

import numpy as np
import pytest
from tinygrad import Tensor, nn
from tinygrad.nn.state import safe_save

from brainchop.tiny_meshnet import load_meshnet

LAYER = {"kernel_size": 1, "padding": 0, "stride": 1, "dilation": 1}


def config(**options):
    """Two 1x1x1 conv layers, 1 -> 1 -> 2 channels; options are top-level config keys."""
    return {"bnorm": False, "dropout_p": 0, **options, "layers": [
        {"in_channels": 1, "out_channels": 1, **LAYER},
        {"in_channels": 1, "out_channels": 2, **LAYER},
    ]}


def weight(*values):
    """A 1x1x1 conv weight with one input channel and one output channel per value."""
    return Tensor(np.array(values, dtype=np.float32).reshape(-1, 1, 1, 1, 1))


def vector(*values):
    return Tensor(np.array(values, dtype=np.float32))


def load(tmp_path, config, state):
    """Write the config and a real safetensors checkpoint, then load them."""
    config_fn, model_fn = tmp_path / "model.json", tmp_path / "model.pth"
    config_fn.write_text(json.dumps(config))
    safe_save(state, str(model_fn))
    return load_meshnet(str(config_fn), str(model_fn))


def test_undeclared_conv_biases_raise(tmp_path):
    state = {"model.0.weight": weight(2), "model.0.bias": vector(0.5),
             "model.2.weight": weight(0, 1), "model.2.bias": vector(1.5, -1.5)}
    with pytest.raises(ValueError, match=r"model\.0\.bias.*model\.2\.bias"):
        load(tmp_path, config(), state)


def test_undeclared_groupnorm_affine_raises(tmp_path):
    state = {"model.0.weight": weight(2), "model.1.weight": vector(3), "model.1.bias": vector(0.25),
             "model.3.weight": weight(0, 1)}
    with pytest.raises(ValueError, match=r"model\.1\.weight.*model\.1\.bias"):
        load(tmp_path, config(bnorm=True), state)


# Unlike the torch path, a lone head bias is not restored here: it has to be declared.
@pytest.mark.parametrize("head", [{"bias": False}, {}], ids=["bias-false", "bias-undeclared"])
def test_undeclared_head_bias_raises(tmp_path, head):
    cfg = config()
    cfg["layers"][-1].update(head)
    state = {"model.0.weight": weight(2),
             "model.2.weight": weight(0, 1), "model.2.bias": vector(1.5, -1.5)}
    with pytest.raises(ValueError, match=r"model\.2\.bias"):
        load(tmp_path, cfg, state)


def test_declared_checkpoint_loads_every_tensor(tmp_path):
    state = {"model.0.weight": weight(2), "model.0.bias": vector(0.5),
             "model.1.weight": vector(3), "model.1.bias": vector(0.25),
             "model.3.weight": weight(0, 1), "model.3.bias": vector(1.5, -1.5)}
    model = load(tmp_path, config(bnorm=True, affine=True, bias=True), state)
    loaded = nn.state.get_state_dict(model)
    for name, value in state.items():
        np.testing.assert_array_equal(loaded[name].numpy(), value.numpy())

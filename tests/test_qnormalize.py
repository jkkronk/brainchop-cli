"""Quantile normalization windows each volume of a batch as if it were alone."""
import numpy as np
import pytest
from tinygrad import Tensor

from brainchop.tiny_meshnet import qnormalize


def volume(seed, scale=1.0):
    """A tiny (1, D, H, W) volume with a skewed, MRI-like intensity histogram."""
    return (scale * np.random.default_rng(seed).gamma(2.0, 20.0, (1, 6, 7, 8))).astype(np.float32)


@pytest.mark.parametrize("shape", [(1, 1, 6, 7, 8), (6, 7, 8)], ids=["single-volume", "3d"])
def test_unbatched_input_matches_whole_tensor_formula(shape):
    img = volume(0).reshape(shape)
    qlow, qhigh = np.quantile(img, 0.02), np.quantile(img, 0.98)
    expected = np.clip((img - qlow) / (qhigh - qlow + 1e-3), 0, 1)
    result = qnormalize(Tensor(img)).numpy()
    assert result.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))


# A batch-mate from the same intensity range, and one from a very different range.
@pytest.mark.parametrize("scale", [1, 40], ids=["same-range", "different-range"])
def test_batched_volumes_are_normalized_as_if_alone(scale):
    vols = [volume(1), volume(2, scale)]
    batched = qnormalize(Tensor(np.stack(vols))).numpy()
    for vol, result in zip(vols, batched):
        np.testing.assert_array_equal(result, qnormalize(Tensor(vol[None])).numpy()[0])

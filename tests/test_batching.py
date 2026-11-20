"""Batching must not change results: a volume gets the same labels alone or in a shard."""

from pathlib import Path

import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.helpers import fetch

from brainchop import Volume, load, segment

TEST_URL = "https://github.com/neuroneural/brainchop-models/raw/main/t1_crop.nii.gz"


@pytest.fixture(scope="module")
def volumes() -> list[Volume]:
    scan = load(Path(fetch(TEST_URL, "t1_crop.nii.gz")))
    # Same anatomy, different intensity histogram (gamma 0.5): a normalization
    # window shared across the shard would change both volumes' labels.
    data = scan.data.numpy().astype(np.float32)
    brighter = (255 * np.sqrt(data / 255)).astype(np.uint8)
    return [scan, Volume(Tensor(brighter), scan.header)]


def test_shard_matches_single_volumes(volumes):
    singles = [segment(v, "tissue_fast").data.numpy() for v in volumes]
    sharded = segment(volumes, "tissue_fast", shard_size=2)

    assert len(sharded) == len(volumes)
    for single, result in zip(singles, sharded):
        labels = result.data.numpy()
        assert labels.shape == single.shape == (256, 256, 256)
        # Allow a few voxels for backends whose batched kernels round differently;
        # results that depend on the shard differ in 1-5% of voxels.
        assert np.mean(labels != single) < 1e-5

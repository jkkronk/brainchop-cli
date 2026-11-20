"""
Example: batch processing with brainchop Python API

load() accepts a single path or a list of paths, so you can split
files however you like—by subject, by session, by folder, etc.
"""

from pathlib import Path
from brainchop import load, segment, save

# collect the inputs before writing anything, so the outputs written below
# (and those of earlier runs) are not segmented as if they were scans
niftis = sorted(p for p in Path(".").glob("*.nii.gz") if not p.name.endswith("_seg.nii.gz"))


def seg_name(path: Path) -> str:
    return path.name.removesuffix(".nii.gz") + "_seg.nii.gz"


# single file (same as before)
vol = load("brain.nii.gz")
result = segment(vol, "tissue_fast")
save(result, "single_seg.nii.gz")

# batch: load a list of paths at once
vols = load(niftis)                        # returns list[Volume]
results = segment(vols, "tissue_fast")     # segment all at once
for path, res in zip(niftis, results):
    save(res, seg_name(path))

# batch with GPU sharding
# shard_size controls how many volumes are batched on GPU per step.
# larger shard_size = faster, but uses more VRAM.
results = segment(vols, "tissue_fast", shard_size=2)

# custom split logic
# load() gives you full control over how to group files.
# for example, process subjects in chunks of 4:
chunk_size = 4
for i in range(0, len(niftis), chunk_size):
    chunk = niftis[i : i + chunk_size]
    vols = load(chunk)
    results = segment(vols, "tissue_fast", shard_size=2)
    for path, res in zip(chunk, results):
        save(res, seg_name(path))

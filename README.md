> [!WARNING]
> **This repository (`brainchop-cli`, tinygrad/Python runtime) is deprecated and is no longer the package served on PyPI for `brainchop >= 1.0.0`.**
>
> **Active development and PyPI releases (`brainchop >= 1.0.0`) have moved to [neuroneural/brainchop-core](https://github.com/neuroneural/brainchop-core):**
> - **Zero ML dependencies** — native C/Metal/CUDA/HIP/CPU binaries with 4 embedded v4.0 Gold / v3.5 models (no tinygrad, PyTorch, CUDA Toolkit, or model downloads at runtime).
> - **Drop-in `pip install brainchop` & `uvx brainchop`** (prebuilt wheels for macOS, Linux, Windows + backward-compatible stdlib `import brainchop` API), plus **Homebrew** (`brew install neuroneural/tap/brainchop`) and **npm** (`@brainchop/core`).
> - **Need the legacy tinygrad runtime or retired models (`--custom`, `export`, older checkpoints)?** Pin the final legacy release:
>   ```bash
>   pip install brainchop==0.2.5
>   ```
>
> | Legacy `0.2.5` model | Recommended `brainchop >= 1.0.0` (`brainchop-core`) model |
> | --- | --- |
> | `mindgrab` / `--skull-strip` | `mindgrab` / `--skull-strip` |
> | `subcortical` | `mindmap` (24ch, 18-class v4.0 Gold) or `mindmap-fast` (16ch v3.5) |
> | `tissue_fast` | `mindmap-fast` (18-class) or `mindmap --pve` (CAT-lite GM/WM/CSF) |
> | `DKatlas` | `mindsnap` (24ch, 104-class v4.0 Gold) |

---

# BrainChop

BrainChop is a lightweight tool for brain segmentation that runs on pretty much everything.

---

## (NEW) No install single command use

Using [uv](https://docs.astral.sh/uv/), brainchop can be ran without needing an enviroment
```
uvx brainchop --help
```

```

## Installation

```bash
pip install brainchop
```

For development (includes docs, testing):

```bash
pip install -e ".[all]"
```

## CLI Usage

```bash
# Segment a brain MRI
brainchop input.nii.gz -o output.nii.gz

# List available models
brainchop --list

# Use a specific model
brainchop input.nii.gz -m subcortical -o output.nii.gz

# Skull stripping
brainchop input.nii.gz --skull-strip -o brain.nii.gz

# With BEAM optimization
brainchop input.nii.gz -m tissue_fast --beam 2 -o output.nii.gz
```

## Python API

```python
import brainchop as bc

# List available models
print(bc.list_models())

# Load, segment, save
vol = bc.load("input.nii.gz")
result = bc.segment(vol, "subcortical")
bc.save(result, "output.nii.gz")

# With BEAM optimization
bc.optimize("tissue_fast", beam=2)
result = bc.segment(vol, "tissue_fast")

# Export to WebGPU
bc.export("tissue_fast", "/tmp/export")
```

## Documentation

Serve docs locally:

```bash
mkdocs serve -w brainchop/
```

## Docker

```bash
git clone git@github.com:neuroneural/brainchop-cli.git
cd brainchop-cli
docker build -t brainchop .
```

Then to run:

```bash
docker run --rm -it --device=nvidia.com/gpu=all -v [[output directory]]:/app brainchop [[input nifti file]] -o [[output nifti file]]
```

## Requirements

- Python 3.10+
- tinygrad
- numpy
- requests

to use the WEBGPU export backend, also install dawn
```
brew tap wpmed92/dawn
brew install dawn
```

## License

MIT License

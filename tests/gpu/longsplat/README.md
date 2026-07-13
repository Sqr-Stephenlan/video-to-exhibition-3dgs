# GPU tests for the LongSplat module

These tests require a real LongSplat repository at the locked commit
(`19750775a9d19f30aa05a8333c4c6c231b2d5f4a`) and a CUDA-capable GPU.

## Prerequisites

- CUDA 12.8 toolchain
- `NVIDIA/LongSplat` repo cloned and checked out at the locked commit with all
  submodules initialised
- Python environment with LongSplat dependencies installed
- `plyfile` and `numpy` (already covered by the project venv)

## Running

```bash
# From the project root
./dev.sh pytest tests/gpu/longsplat/ -v
```

## What these tests cover

- End-to-end training on a minimal synthetic dataset (smoke test)
- Converter end-to-end with a real LongSplat checkpoint
- Run record lifecycle across training and conversion stages

These tests are NOT run in CI and must be executed manually on GPU hardware.

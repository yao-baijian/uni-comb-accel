# uni-comb-accel

## 1. Prerequisites

- OS: Linux (validated on Ubuntu-like environments)
- Python: 3.11+
- Git + submodule support
- C/C++ toolchain: `clang`, `lld`, `ninja`, `cmake`
- Python packages: see `requirements.txt`
- Optional (for real hardware build):
  - Xilinx Vitis/Vivado (current local flow uses `2025.1`)
  - Board-specific device support packages (for example VC1902/VM1802 parts)
  - PetaLinux sysroot + `xilinx-versal-common-*` image/rootfs when packaging

## 2. Installation (Project + Subprojects)

### 2.1 Clone and init submodules

```bash
git clone <your-repo-url>
cd uni-comb-accel
git submodule update --init --recursive
```

### 2.2 Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.3 Build ARIES subproject

```bash
cd tools/ARIES
source aries/bin/activate
source /tools/Xilinx/2025.1/Vitis/settings64.sh
source utils/build-llvm.sh
source utils/build-mlir-aie.sh
source utils/build-aries.sh
cd ../..
```

## 3. Latest Progress

- Python -> MLIR -> ARIES backend compile path is wired through `src/api.py`.
- Sparse TCSR path exists (`src/backend/tcsr.py`, `src/backend/sparse_to_aie.py`).
- Autodiff frontend (JAXPR -> MLIR) is integrated (`src/compiler/autodiff.py`).
- Shape-policy compile reuse is implemented:
  - `exact`
  - `bucket`
  - `max_shape`
- New compile tests are added in `tests/compile_test.py`:
  - full compile-flow smoke test (mocked backend)
  - shape hit/recompile cache behavior test
- ARIES example Makefiles are updated to prefer local Vitis installation paths.
- Known current hardware blocker on this machine:
  - Vitis can launch, but board part packages (for example `xcvc1902` / `xcvm1802`) are not fully installed, so `.xsa/.xclbin` generation is blocked at device-part load.

## 4. Simple End-to-End Example

This example uses the stable manual frontend path and emits MLIR + optimized MLIR + AIE C++ code.

```python
import numpy as np
from src.api import compile_energy_function

N = 4

def energy(x):
    return sum(x[i] * x[i] for i in range(N))

x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)

artifacts = compile_energy_function(
    energy,
    example_args=(x,),
    target="aie",
    output_dir="build/real_compile_manual",
    gradient_mode="manual",
    variables={"x": {"shape": (4,)}},
)

print(artifacts)
```

Expected local outputs:

- `build/real_compile_manual/energy.mlir`
- `build/real_compile_manual/energy.opt.mlir`
- `build/real_compile_manual/energy_aie.cc`
- `build/real_compile_manual/energy.aie_config.json`

## Top-Level and First-Level Architecture

```text
uni-comb-accel/
├── src/
│   ├── api.py
│   ├── compiler/
│   ├── backend/
│   ├── runtime/
│   ├── testing/
│   └── sbm/
├── tests/
├── benchmarks/
├── docs/
├── tools/
│   ├── ARIES/
│   └── FEM/
└── build/
```

More detailed sub-architecture docs are in:

- `docs/architecture_overview.md`
- `docs/autodiff.md`
- `docs/aie_model.md`
- `docs/runthrough_changes_2026-04-03.md`

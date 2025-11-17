# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Building and Development Commands

### Initial Setup
```bash
# Clone and setup development environment
git clone git@github.com:<USERNAME>/pytorch.git
cd pytorch
git remote add upstream git@github.com:pytorch/pytorch.git

# Setup development environment with pre-built binaries (recommended for Python-only development)
make setup-env  # CPU-only
# OR for GPU development:
make setup-env-cuda  # CUDA
make setup-env-rocm  # AMD ROCm

source venv/bin/activate
```

### Building from Source
```bash
# Install dependencies first
pip install -r requirements.txt
conda install cmake ninja

# Build PyTorch in development mode
python setup.py develop

# Common build options (set as environment variables):
DEBUG=1                    # Debug build with symbols (-g -O0)
REL_WITH_DEB_INFO=1       # Optimized build with debug symbols (-g -O3)
USE_CUDA=0                # Disable CUDA
USE_DISTRIBUTED=0         # Disable distributed training
BUILD_TEST=0              # Skip building C++ tests
MAX_JOBS=8                # Parallel compilation jobs

# Example: Fast CPU-only debug build
DEBUG=1 USE_CUDA=0 USE_DISTRIBUTED=0 BUILD_TEST=0 python setup.py develop

# After initial build, rebuild specific components:
cd build && ninja torch_cpu  # Rebuild just CPU library
cd build && ninja test_jit   # Rebuild specific test binary
```

### Testing

**Python Tests:**
```bash
# Run all tests
python test/run_test.py

# Run specific test file
python test/test_nn.py

# Run specific test class/method
python test/test_jit.py TestJit.test_Sequential

# Run tests matching substring with pytest
pytest test/test_nn.py -k Loss -v
```

**C++ Tests:**
```bash
# C++ test binaries are in build/bin/
./build/bin/test_jit --gtest_filter=ContainerAliasingTest.MayContainAlias
```

### Linting and Code Quality
```bash
make lint          # Run all linters (flake8, mypy, clang-tidy, etc.)
make quicklint     # Lint only changed files
make quickfix      # Auto-fix linting issues

# Run specific linter on changed files
flake8 $(git diff --name-only $(git merge-base --fork-point main))
```

### Building Documentation
```bash
cd docs/
pip install -r requirements.txt
make html        # Build HTML docs
make serve       # Serve locally at localhost:8000
make doctest     # Run documentation tests
```

### Development Workflow Tips
```bash
# Clean build (if you encounter issues)
python setup.py clean
rm -rf build

# Reinstall after modifying C++/CUDA files
python setup.py develop

# For Python-only changes, no reinstall needed (using develop mode)

# Debug C++ with symbols (rebuild specific file with debug info)
./tools/build_with_debinfo.py torch/csrc/autograd/python_variable_indexing.cpp
```

## Code Architecture

### Layered Architecture (Dependency Flow)

```
torch/              # Python API layer (nn.Module, optim, etc.)
  ↓
torch/csrc/         # Python-C++ bindings (pybind11)
  ↓
aten/               # ATen: C++ tensor library (operators, kernels)
  ↓
c10/                # Core utilities (no PyTorch dependencies, standalone)
```

**Key Principle:** Dependencies flow downward. c10 has no dependencies on ATen or torch. ATen doesn't know about Python.

### Key Directories

**c10/** - Core foundational library
- Device abstraction (CPU/CUDA/MPS/XPU)
- Type system (`ScalarType.h`, `DType.h`)
- Dispatch mechanism (`DispatchKey.h`, `DispatchKeySet.h`)
- Tensor metadata (`TensorImpl.h`, `Storage.h`)
- Low-level utilities (threading, exceptions, intrusive_ptr)
- **Purpose:** Reusable core that works everywhere (mobile, server, embedded)

**aten/** - ATen tensor library
- `aten/src/ATen/native/` - Operator implementations (CPU, CUDA, MPS)
  - `native_functions.yaml` - **CENTRAL REGISTRY** of all tensor operations
  - `BinaryOps.cpp`, `Activation.cpp`, etc. - Operator implementations
  - `cpu/`, `cuda/`, `mps/`, `mkldnn/` - Backend-specific kernels
- `aten/src/ATen/core/` - Core abstractions (migrating to c10/)
- **Pattern:** Each operator typically has multiple backend implementations

**torch/** - Python package
- Pure Python modules: `nn/`, `optim/`, `utils/`
- `_C/` - Type stubs for C++ extensions
- `_dynamo/` - Dynamic shape tracing and JIT compilation
- `_inductor/` - Graph compilation backend
- `fx/` - Function transformation framework
- `jit/` - TorchScript Python API

**torch/csrc/** - Python-C++ bridge
- `Module.cpp` - Main Python extension module initialization
- `autograd/` - Autograd engine implementation (gradient computation)
- `jit/` - TorchScript compiler (frontend, IR, optimizer passes, runtime)
- `api/` - C++ frontend (libtorch)
- `distributed/` - Distributed training (c10d, gloo, nccl)
- **Pattern:** Uses pybind11 to expose C++ to Python; all exports marked with `*_API` macros

**tools/** - Build and code generation
- `torchgen/` - **Code generation engine** (reads YAML, generates C++ dispatch code)
  - `gen.py` - Main orchestrator for code generation
  - `gen_backend_stubs.py`, `gen_autograd.py`, etc.
- `autograd/` - Derivative rule definitions
- `build_with_debinfo.py` - Helper to rebuild specific files with debug symbols

**test/** - Testing infrastructure
- `test_*.py` - Unit tests for Python modules
- `cpp/` - C++ unit tests (uses GoogleTest)
- `expect/` - Expected output files for comparison
- Run from test directory in CI (not repo root)

### Code Generation Pipeline (Critical System)

PyTorch heavily uses code generation to maintain consistency across backends:

```
native_functions.yaml
  (defines all operators: signatures, backends, dispatch rules)
        ↓
    torchgen/ (Python code generator)
        ├─→ Dispatch routing code
        ├─→ Autograd (derivative) formulas
        ├─→ Python bindings
        ├─→ Functionalization wrappers
        └─→ Backend stubs
        ↓
Generated files in:
  - build/aten/src/ATen/
  - build/torch/csrc/
```

**Key File:** `aten/src/ATen/native/native_functions.yaml`
- Declarative definition of all tensor operations
- Example entry:
  ```yaml
  - func: add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    variants: function, method
    dispatch:
      CPU: add_cpu
      CUDA: add_cuda
    tags: pointwise
  ```

**Why code generation?**
- Ensures backend consistency (CPU, CUDA, MPS, etc. all support same ops)
- Automatic derivative rules
- Type-safe dispatch mechanism
- Reduces boilerplate

### Dispatch Mechanism (Runtime Routing)

Operations are routed through a sophisticated dispatch system:

```
torch.add(x, y)
     ↓
Dispatcher (c10/core/impl/LocalDispatchKeySet.h)
     ↓
Selects backend based on DispatchKeySet:
  - CPU → cpu kernel
  - CUDA → cuda kernel
  - Autograd → gradient recording
  - Functionalize → convert in-place to functional
  - Lazy → deferred execution
  - [30+ dispatch keys total]
```

**Key Implementation Files:**
- `c10/core/DispatchKey.h` - Enum of all dispatch keys
- `c10/core/impl/TorchDispatchModeTLS.h` - Thread-local dispatch configuration
- Operators register implementations via `TORCH_LIBRARY_IMPL` macro

### Python-C++ Integration Pattern

```python
# Python (torch/nn/functional.py)
def linear(input, weight, bias=None):
    return torch._C._nn.linear(input, weight, bias)
         ↓
# C++ Binding (torch/csrc/nn/init.cpp)
TORCH_LIBRARY_IMPL(nn, CPU, m) {
  m.impl("linear", &at::linear);
}
         ↓
# ATen Implementation (aten/src/ATen/native/Linear.cpp)
Tensor linear(const Tensor& input, const Tensor& weight, ...) {
  // Backend-specific implementation
}
```

**Memory Management:**
- Python side: Reference counting via `Py_INCREF`/`Py_DECREF`
- C++ side: `intrusive_ptr` for tensors (in `c10/util/intrusive_ptr.h`)
- Bridge: Careful GIL (Global Interpreter Lock) management

### Multi-Backend Architecture

Every operation can have implementations for:
- **CPU:** Standard implementations + vectorized (AVX/NEON) in `aten/src/ATen/cpu/vec/`
- **CUDA:** GPU kernels in `aten/src/ATen/native/cuda/`
- **MPS:** Metal Performance Shaders (Apple GPU) in `aten/src/ATen/native/mps/`
- **XPU:** Intel GPU in `aten/src/ATen/native/xpu/`
- **ROCm/HIP:** AMD GPU (transpiled from CUDA via HIPify)
- **Vulkan:** Cross-platform graphics API

**Pattern:** Same operator name, multiple backend implementations, routed via dispatch.

## Common Development Patterns

### Adding a New Operator

1. **Define in YAML** (`aten/src/ATen/native/native_functions.yaml`):
   ```yaml
   - func: my_op(Tensor self, Scalar value) -> Tensor
     variants: function, method
     dispatch:
       CPU: my_op_cpu
       CUDA: my_op_cuda
   ```

2. **Implement backends:**
   - CPU: `aten/src/ATen/native/MyOp.cpp`
   - CUDA: `aten/src/ATen/native/cuda/MyOp.cu`

3. **Add derivative rule** (if differentiable) in `tools/autograd/derivatives.yaml`

4. **Rebuild:** Code generation runs automatically during build

### Modifying Autograd Behavior

- **Derivative formulas:** `tools/autograd/derivatives.yaml`
- **Autograd engine:** `torch/csrc/autograd/` (engine.cpp, functions/)
- **Custom autograd functions:** Subclass `torch.autograd.Function`

### Working with TorchScript/JIT

- **Frontend (Python → IR):** `torch/csrc/jit/frontend/`
- **IR (Intermediate Representation):** `torch/csrc/jit/ir/`
- **Optimization passes:** `torch/csrc/jit/passes/`
- **Runtime interpreter:** `torch/csrc/jit/runtime/`

### Debugging Dispatch

Set environment variables to trace dispatch routing:
```bash
TORCH_SHOW_DISPATCH_TRACE=1 python your_script.py
```

This requires building with: `CFLAGS="-DHAS_TORCH_SHOW_DISPATCH_TRACE" python setup.py develop`

## Important Configuration

### Build-Time Feature Flags (Environment Variables)

Set before running `setup.py develop`:
```bash
USE_CUDA=0/1              # CUDA support
USE_CUDNN=0/1             # cuDNN acceleration
USE_DISTRIBUTED=0/1       # Distributed training (c10d, gloo, mpi)
USE_MKLDNN=0/1            # Intel MKL-DNN
USE_FBGEMM=0/1            # Quantized operators
USE_FLASH_ATTENTION=0/1   # Flash attention kernels
USE_XNNPACK=0/1           # Mobile optimized kernels
BUILD_TEST=0/1            # Build C++ tests
DEBUG=1                   # Debug symbols
MAX_JOBS=N                # Parallel compilation jobs
```

### Runtime Feature Toggles

Environment variables affecting runtime behavior:
- `TORCH_SHOW_CPP_STACKTRACES=1` - Show C++ stack traces in Python errors
- `TORCH_COMPILE_DEBUG=1` - Debug torch.compile
- `OMP_NUM_THREADS=N` - OpenMP thread count
- `CUDA_LAUNCH_BLOCKING=1` - Synchronous CUDA execution (debugging)

## Performance Optimization

### Fast Incremental Builds

1. **Use Ninja:** `pip install ninja` (automatically detected)
2. **Use ccache:** `sudo apt install ccache` (speeds up recompilation)
3. **Use faster linker:** `CMAKE_LINKER_TYPE=MOLD python setup.py develop` (requires CMake 3.29+)
4. **Pre-compiled headers:** `USE_PRECOMPILED_HEADERS=1 python setup.py develop`

### Debugging Performance

```bash
# Build with profiling support
python setup.py develop

# Profile with py-spy (samples Python + C++ stacks)
pip install py-spy
py-spy record -o profile.svg --native -- python your_script.py
```

### CUDA Development

```bash
# Enable CUDA debug symbols
CUDA_DEVICE_DEBUG=1 python setup.py develop

# Use cuda-gdb instead of gdb
cuda-gdb python

# Use cuda-memcheck for memory errors
cuda-memcheck python your_script.py
```

## Testing Best Practices

### Writing Tests

- Python tests go in `test/test_*.py`
- C++ tests go in `test/cpp/`
- Follow existing test patterns (see `test/test_torch.py` for examples)
- Use `torch.testing.assert_close()` for numerical comparisons (handles floating point tolerance)

### CI and Pull Requests

- CI runs on merge of PR branch with `main` (uses workflow files from merge commit)
- Check [HUD](https://hud.pytorch.org) for CI status
- If CI fails on `main`, your PR failure may be unrelated
- Use `@pytorchmergebot merge` to merge approved PRs

### Common Test Patterns

```python
# Test decorator for device-specific tests
@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
def test_my_cuda_function(self):
    ...

# Test all dtypes
from torch.testing._internal.common_dtype import all_types_and_complex
@dtypes(*all_types_and_complex())
def test_op(self, dtype):
    ...

# Instantiate tests for different devices
instantiate_device_type_tests(TestMyClass, globals())
```

## Special Considerations

### Symbol Visibility (Windows/MSVC)

- Symbols must be explicitly exported: use `TORCH_API`, `CAFFE2_API` macros
- Example: `TORCH_API Tensor my_function(const Tensor& x);`
- Template functions should NOT use `*_API` (instantiated at call site)

### Memory Safety

- PyTorch uses **view semantics**: operations like `reshape()`, `transpose()` don't copy
- Views share underlying storage (tracked in `TensorImpl`)
- Be careful with in-place operations on views
- Use `contiguous()` if you need a contiguous copy

### Submodules

```bash
# After pulling changes, update submodules
git submodule sync
git submodule update --init --recursive
```

### Caffe2 Legacy

- `caffe2/` directory contains legacy Caffe2 codebase
- Mostly deprecated; PyTorch is primary focus
- Kept for backward compatibility

## Key Files to Understand

1. **`aten/src/ATen/native/native_functions.yaml`** - Central operator registry
2. **`torch/csrc/Module.cpp`** - Python extension initialization
3. **`c10/core/DispatchKey.h`** - Dispatch mechanism
4. **`tools/autograd/derivatives.yaml`** - Gradient rules
5. **`torchgen/gen.py`** - Code generation orchestrator
6. **`torch/csrc/autograd/engine.cpp`** - Autograd backward pass
7. **`aten/src/ATen/core/TensorBody.h`** - Tensor class definition

## Additional Resources

- **Contributing Guide:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Developer Wiki:** https://github.com/pytorch/pytorch/wiki
- **PyTorch Forums:** https://discuss.pytorch.org
- **HUD (CI Status):** https://hud.pytorch.org
- **Dev Infra Office Hours:** Every Friday (see Wiki for details)

# CuteDSL Blackwell GPU Support - Implementation Plan

## Problem Statement

The "Arch conditional MMA instruction used without targeting appropriate compute capability" error occurs in an infinite loop when PyTorch tries to compile CuteDSL kernels for Blackwell GPUs (compute capability 9.0/10.0). This prevents the use of torch._grouped_mm and other CuteDSL operations on Blackwell hardware.

## Root Cause

The @cute.kernel decorator compiles CUDA kernels using NVRTC at module load time. When PyCodeCache loads a CuteDSL module, the compilation happens immediately without the proper architecture flags (-arch=sm_90, -arch=sm_100), causing NVRTC to fail when it encounters Blackwell-specific MMA instructions.

## Implemented Fixes

### 1. Critical Fix: Set Environment Variable Before Module Load
**File**: `torch/_inductor/async_compile.py`
**Commit**: 54725ee55e4

```python
def cutedsl(self, kernel_name: str, source_code: str):
    def task():
        # CRITICAL: Set CUTLASS_NVCC_ARCHS before loading the module
        import os
        if "CUTLASS_NVCC_ARCHS" not in os.environ:
            os.environ["CUTLASS_NVCC_ARCHS"] = "90a-real,100a-real"

        key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
        mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)
```

This ensures the environment variable is set BEFORE the module is executed, which is when @cute.kernel decorators trigger compilation.

### 2. Module-Level Configuration
**File**: `torch/_inductor/codegen/cutedsl/__init__.py`
**Commit**: b9284fcb198

```python
import os
# Configure CUTLASS NVCC architecture BEFORE any cutlass imports
if "CUTLASS_NVCC_ARCHS" not in os.environ:
    os.environ["CUTLASS_NVCC_ARCHS"] = "90a-real,100a-real"
```

Sets the environment variable at the earliest possible point in the module import chain.

### 3. Import Order Fixes in All CuteDSL Files

#### a. Template Files
**File**: `torch/_inductor/kernel/templates/cutedsl_mm_grouped.py.jinja`
**Commit**: d036c19cab4

#### b. Vendored Templates
**File**: `torch/_inductor/kernel/vendored_templates/cutedsl_grouped_gemm.py`
**Commit**: 5d07270a732

#### c. Utils Module
**File**: `torch/_inductor/codegen/cutedsl/_cutedsl_utils.py`
**Commit**: 5d07270a732

#### d. Kernel Generation
**File**: `torch/_inductor/codegen/cutedsl/cutedsl_kernel.py`
**Commit**: 5d07270a732

All these files now set CUTLASS_NVCC_ARCHS before importing cutlass modules.

### 4. Supporting Fixes

#### a. Compute Capability Check
**File**: `aten/src/ATen/native/cuda/Blas.cpp`
**Commit**: 5fe88b3242b

Changed from `== 9` to `>= 9` to support Blackwell 10.0+:
```cpp
if (sm90_only) {
  return dprops->major >= 9;  // Support 9.0 and 10.0+
}
```

#### b. Missing cutedsl() Method
**File**: `torch/_inductor/async_compile.py`
**Commit**: d3926ef4b5a

Added the missing cutedsl() method to AsyncCompile class.

#### c. Test Fixes
**Files**: `test_grouped_mm.py`, `test_cutedsl_arch.py`
**Commits**: f31da31d6b4, bc5205e6072

Fixed:
- Stride alignment (multiples of 16 bytes)
- Offset dtype (int32 instead of int64)
- HardwareInfo API compatibility

## Verification Steps

### 1. Clear All Caches
```bash
rm -rf ~/.cache/torch ~/.triton/cache
```

### 2. Run Test Suite
```bash
# Basic environment variable test
python test_cutedsl_env_var.py

# Minimal CUTLASS test
python test_cutedsl_minimal.py

# Full grouped_mm test
python test_grouped_mm.py

# Architecture detection test
python test_cutedsl_arch.py
```

### 3. Build New Wheel
```bash
python setup.py bdist_wheel
```

## Deployment Checklist

- [ ] All tests pass locally
- [ ] PyCodeCache cleared
- [ ] New wheel built with all fixes
- [ ] Wheel deployed to test environment
- [ ] Bolt task runs without MMA instruction errors
- [ ] torch._grouped_mm operations complete successfully
- [ ] Performance metrics collected

## Alternative Solutions (If Primary Fix Fails)

### 1. System-Level Environment Variable
Set before Python starts:
```bash
export CUTLASS_NVCC_ARCHS="90a-real,100a-real"
python your_script.py
```

### 2. Direct NVRTC Flag Injection
Modify kernel compilation to add flags directly to NVRTC calls.

### 3. Build-Time Configuration
Configure CUTLASS architectures in CMake during PyTorch build.

## Key Insights

1. **Timing is Critical**: Environment variables must be set BEFORE module execution, not just before cutlass import.

2. **PyCodeCache Execution**: When PyCodeCache loads a module, it immediately executes the module code, including decorators.

3. **Decorator Compilation**: @cute.kernel decorators compile kernels immediately when applied, not when the kernel is called.

4. **Multi-Process Considerations**: Fixes must work in both single-threaded and multi-process compilation modes.

## Expected Outcome

With all fixes applied:
1. No "Arch conditional MMA instruction" errors
2. torch._grouped_mm works on Blackwell GPUs
3. CuteDSL kernels compile with proper architecture flags
4. vLLM and other frameworks can use CuteDSL on Blackwell

## Commit History

```
54725ee55e4 Fix CuteDSL compilation by setting CUTLASS_NVCC_ARCHS before module load
0d3174e1864 Add __init__.py check to CuteDSL environment variable validation test
b9284fcb198 Add CUTLASS_NVCC_ARCHS to CuteDSL __init__.py for earliest possible configuration
026169e184e Add comprehensive test documentation for CuteDSL grouped GEMM
f31da31d6b4 Fix stride alignment issues in test_grouped_mm.py
bc5205e6072 Fix HardwareInfo API usage and add torch._grouped_mm test script
633dc996148 Add tests to validate CUTLASS architecture configuration
d036c19cab4 Fix import order in CuteDSL template to set CUTLASS_NVCC_ARCHS before cutlass import
5d07270a732 Add CUTLASS NVCC architecture configuration to all CuteDSL import paths
a8b2a074038 Configure CUTLASS NVCC architecture for Blackwell in CuteDSL template
d3926ef4b5a Add missing cutedsl() method to AsyncCompile
ac71c53b27c Add vendored CuteDSL grouped GEMM template from CUTLASS
5fe88b3242b Fix compute capability check for _grouped_mm to support Blackwell 10.0+
```

## Testing Matrix

| Test | Purpose | Expected Result |
|------|---------|-----------------|
| test_cutedsl_env_var.py | Verify env var is set before imports | All checks pass |
| test_cutedsl_minimal.py | Test basic @cute.kernel compilation | No MMA errors |
| test_grouped_mm.py | Test torch._grouped_mm operations | All 3 tests pass |
| test_cutedsl_arch.py | Test architecture detection | Detects Blackwell correctly |

## Final Notes

The most critical fix is in `torch/_inductor/async_compile.py` where we ensure CUTLASS_NVCC_ARCHS is set immediately before PyCodeCache loads the module. This guarantees the environment variable is available when @cute.kernel decorators compile their kernels.

All other fixes provide defense-in-depth to handle various import paths and edge cases.
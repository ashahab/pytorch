# CuteDSL Grouped GEMM Test Scripts

This directory contains test scripts to validate the CuteDSL architecture configuration fix for Blackwell GPUs.

## Issue Background

The "Arch conditional MMA instruction used without targeting appropriate compute capability" error occurs when nvidia-cutlass JIT-compiles CUDA kernels without proper architecture flags. This happens specifically on Blackwell 10.0 GPUs when `CUTLASS_NVCC_ARCHS` environment variable is not set before importing cutlass.

## Test Scripts

### 1. test_cutedsl_env_var.py (Quick Validation)

**No dependencies required** - validates code configuration without running kernels.

```bash
python test_cutedsl_env_var.py
```

**What it checks:**
- ✅ `CUTLASS_NVCC_ARCHS` is set before `import cutlass` in all files
- ✅ Correct import order in template files
- ✅ Proper configuration in generated code

**Use when:** You want to quickly verify the fix is applied correctly without needing nvidia-cutlass installed.

### 2. test_cutedsl_arch.py (Module Loading)

**Requires:** nvidia-cutlass, CUDA

```bash
python test_cutedsl_arch.py
```

**What it tests:**
- ✅ CuteDSL utility module imports successfully
- ✅ Vendored template imports successfully
- ✅ GPU detection works correctly
- ✅ Simple `@cute.kernel` compilation works (key test!)

**Use when:** You want to verify that cutlass module initialization works correctly with the architecture configuration.

### 3. test_grouped_mm.py (Full Integration)

**Requires:** nvidia-cutlass, CUDA, Blackwell GPU

```bash
# Clear cache first (important!)
rm -rf ~/.cache/torch_inductor/

# Run tests
python test_grouped_mm.py
```

**What it tests:**
- ✅ `torch._grouped_mm` basic functionality
- ✅ Larger matrix dimensions
- ✅ **torch.compile with Inductor** (tests CuteDSL code generation!)

**Use when:** You want to validate the complete end-to-end functionality including code generation and kernel execution.

## Alignment Requirements

**IMPORTANT:** CuteDSL grouped GEMM kernels have strict alignment requirements:

- **Contiguous dimension must be ≥ 16 bytes**
- For `bfloat16` (2 bytes/element): minimum 8 elements
- **Recommended:** Use multiples of 16 for all dimensions

**Example valid dimensions:**
```python
# ✅ Good - multiples of 16
mat_a = torch.randn(48, 16, dtype=torch.bfloat16, device='cuda')
mat_b = torch.randn(2, 16, 16, dtype=torch.bfloat16, device='cuda')

# ❌ Bad - contiguous dim too small
mat_a = torch.randn(12, 8, dtype=torch.bfloat16, device='cuda')   # 8 < 16 bytes
mat_b = torch.randn(2, 8, 4, dtype=torch.bfloat16, device='cuda')  # 4 < 8 elements
```

## Expected Results

With the fix applied:

```
✅ Test 1: Simple grouped_mm - PASS
✅ Test 2: Larger grouped_mm - PASS
✅ Test 3: torch.compile grouped_mm - PASS

✓ ALL TESTS PASSED
The torch._grouped_mm operation is working correctly!
The 'Arch conditional MMA instruction' fix is successful.
```

## Troubleshooting

### "Arch conditional MMA instruction" error

**Still seeing this error?** The fix isn't working. Check:

1. **Clear the cache:**
   ```bash
   rm -rf ~/.cache/torch_inductor/
   # or
   export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
   ```

2. **Verify fix is applied:**
   ```bash
   python test_cutedsl_env_var.py  # Should show all ✓
   ```

3. **Check CUTLASS_NVCC_ARCHS at runtime:**
   ```python
   import os
   print(os.environ.get('CUTLASS_NVCC_ARCHS'))
   # Should print: 90a-real,100a-real
   ```

### "strides should be multiple of 16 bytes" error

Your tensor dimensions are not properly aligned. See "Alignment Requirements" above.

### "compute capability = 9.0" error on SM 10.0 GPU

The C++ fix for compute capability checking hasn't been applied. Rebuild PyTorch:
```bash
python setup.py develop
```

### Circular import error in test_cutedsl_arch.py

This is a known PyTorch v2.8.0 issue unrelated to the fix. The test handles it gracefully.

## The Fix

The fix ensures `CUTLASS_NVCC_ARCHS="90a-real,100a-real"` is set **before** any cutlass imports in:

1. `torch/_inductor/kernel/templates/cutedsl_mm_grouped.py.jinja`
2. `torch/_inductor/kernel/vendored_templates/cutedsl_grouped_gemm.py`
3. `torch/_inductor/codegen/cutedsl/_cutedsl_utils.py`
4. `torch/_inductor/codegen/cutedsl/cutedsl_kernel.py` (gen_imports)
5. `aten/src/ATen/native/cuda/Blas.cpp` (>= 9 instead of == 9)

This tells nvidia-cutlass to compile kernels for:
- **90a**: Blackwell 9.0 with advanced features
- **100a**: Blackwell 10.0+ with advanced features
- **-real**: Native code compilation (not just PTX)

## Quick Start

```bash
# 1. Validate the fix is applied
python test_cutedsl_env_var.py

# 2. Clear cache (critical!)
rm -rf ~/.cache/torch_inductor/

# 3. Run full tests
python test_grouped_mm.py
```

If all tests pass, the fix is working correctly! 🎉

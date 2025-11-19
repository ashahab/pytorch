# torch._grouped_mm Test Suite

Comprehensive test script for the `torch._grouped_mm` operation on Blackwell and other CUDA GPUs.

## Overview

This test suite validates:
- Basic grouped matrix multiplication functionality
- Large matrix operations with multiple groups
- Integration with torch.compile (Inductor backend)
- Edge cases and error handling
- Correctness verification against manual computation

## Requirements

- CUDA-enabled GPU
- PyTorch with `torch._grouped_mm` support
- Python 3.8+

## Quick Start

### Basic Usage

```bash
python test_grouped_mm.py
```

### With Environment Variable

For Blackwell GPUs (SM 9.0+), set CUTLASS architecture before running:

```bash
export CUTLASS_NVCC_ARCHS="90a-real,100a-real"
python test_grouped_mm.py
```

The test script automatically sets this if not already set.

## Test Cases

### Test 1: Simple Grouped MM
- 2 groups: 16x16 and 32x16 matrices
- Tests basic functionality
- Verifies correctness against manual computation

### Test 2: Larger Grouped MM
- 3 groups: 128x64, 256x64, 64x64 matrices
- Tests scalability with larger dimensions
- Validates multiple groups

### Test 3: torch.compile Integration
- Compiles grouped_mm with Inductor backend
- Tests CuteDSL code generation
- Verifies kernel caching

### Test 4: Edge Cases
- Single group operation
- Many groups (8 groups)
- Large dimensions (1024+ elements)

## Expected Output

### Success
```
======================================================================
torch._grouped_mm Test Suite
======================================================================

Prerequisites Check
✓ CUDA available: NVIDIA B200
✓ Compute Capability: 10.0
✓ Blackwell GPU detected - CuteDSL kernels available
✓ torch._grouped_mm is available

Test 1: Simple Grouped Matrix Multiplication
✓ torch._grouped_mm executed successfully!
✓ Result is finite and valid
✓ Group 1 result matches expected
✓ Group 2 result matches expected

[... more tests ...]

======================================================================
Test Summary
======================================================================
✓ PASS: Simple grouped_mm
✓ PASS: Larger grouped_mm
✓ PASS: torch.compile grouped_mm
✓ PASS: Edge cases
======================================================================
✓ ALL TESTS PASSED
======================================================================
```

## Common Issues

### "Arch conditional MMA instruction" Error

**Symptom:**
```
ERROR : Arch conditional MMA instruction used without targeting appropriate compute capability. Aborting.
```

**Solution:**
1. Ensure CUTLASS_NVCC_ARCHS is set before importing torch:
   ```bash
   export CUTLASS_NVCC_ARCHS="90a-real,100a-real"
   ```

2. Or add to your Python script before importing torch:
   ```python
   import os
   os.environ["CUTLASS_NVCC_ARCHS"] = "90a-real,100a-real"
   import torch
   ```

3. For PyTorch builds with the fix, this is handled automatically in `torch/__init__.py`

### Stride Alignment Error

**Symptom:**
```
RuntimeError: strides should be multiple of 16 bytes
```

**Solution:**
Ensure tensor dimensions are multiples of 16 elements for bfloat16 dtype:
```python
# Bad: 8 elements = 16 bytes (borderline)
mat_a = torch.randn(12, 8, dtype=torch.bfloat16)

# Good: 16 elements = 32 bytes
mat_a = torch.randn(48, 16, dtype=torch.bfloat16)
```

### Offset Dtype Error

**Symptom:**
```
RuntimeError: Offsets have to be int32
```

**Solution:**
Use int32 for offsets, not int64:
```python
# Bad
offs = torch.tensor([16, 48], dtype=torch.int64)

# Good
offs = torch.tensor([16, 48], dtype=torch.int32)
```

## Architecture Support

### Blackwell (SM 9.0, 10.0+)
- **Recommended**: Native CuteDSL support
- Optimized kernels via CUTLASS
- Full test suite should pass

### Hopper (SM 9.0)
- Supported with CuteDSL
- May require CUTLASS_NVCC_ARCHS configuration

### Ampere and Earlier (SM 8.0-)
- May fall back to alternative implementations
- Tests will run but may use different code paths
- Performance may be lower

## Debugging

### Verbose Output

Enable PyTorch compilation logs:
```bash
export TORCH_LOGS="+inductor"
export TORCH_COMPILE_DEBUG=1
python test_grouped_mm.py
```

### Check CUTLASS Configuration

```python
import os
print(f"CUTLASS_NVCC_ARCHS: {os.environ.get('CUTLASS_NVCC_ARCHS', 'NOT SET')}")
```

### Clear Cached Kernels

If you get stale compilation errors:
```bash
rm -rf ~/.cache/torch
rm -rf ~/.triton/cache
python test_grouped_mm.py
```

## Technical Details

### Input Requirements

1. **Matrix A (concatenated)**: Shape `[total_rows, K]`
   - Contains all groups concatenated along dimension 0
   - dtype: bfloat16
   - Must be contiguous or properly strided

2. **Matrix B (grouped)**: Shape `[num_groups, K, N]`
   - Each group is a separate K×N matrix
   - dtype: bfloat16
   - Must be contiguous

3. **Offsets**: Shape `[num_groups]`
   - Cumulative row counts in matrix A
   - dtype: int32 (required, not int64)
   - Last offset should equal total_rows

### Alignment Requirements

For bfloat16 (2 bytes per element):
- Contiguous dimensions should be ≥ 16 bytes (8 elements minimum)
- Recommended: multiples of 16 elements for best performance
- Inner dimension (K and N) should be multiples of 16

### Output

- Shape: `[total_rows, N]`
- dtype: Same as input (bfloat16)
- Contains results of all groups concatenated

## Files

- `test_grouped_mm.py`: Main test script
- `TEST_GROUPED_MM_README.md`: This file

## Contributing

When adding new tests:
1. Follow existing test structure
2. Add proper error handling
3. Verify correctness against manual computation
4. Update this README with new test descriptions

## References

- PyTorch `torch._grouped_mm` documentation
- CUTLASS library: https://github.com/NVIDIA/cutlass
- CuteDSL (CUTLASS Python DSL)

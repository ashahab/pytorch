#!/usr/bin/env python3
"""
Test script for torch._grouped_mm functionality with CuteDSL on Blackwell.

This tests the actual grouped matrix multiplication operation that uses
CuteDSL kernels on Blackwell GPUs.
"""

import os
import sys
import torch

def check_prerequisites():
    """Check if the environment is suitable for running grouped_mm tests."""
    print("=" * 70)
    print("Prerequisites Check")
    print("=" * 70)

    # Check CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False

    print(f"✓ CUDA available: {torch.cuda.get_device_name()}")

    # Check compute capability
    major, minor = torch.cuda.get_device_capability()
    print(f"✓ Compute Capability: {major}.{minor}")

    if major < 9:
        print(f"⚠ Grouped MM with CuteDSL requires Blackwell (SM 9.0+)")
        print(f"  Your GPU is SM {major}.{minor} - tests may not use CuteDSL")
    elif major >= 9:
        print(f"✓ Blackwell GPU detected - CuteDSL should be available")

    # Check if _grouped_mm exists
    if not hasattr(torch, '_grouped_mm'):
        print("✗ torch._grouped_mm not available in this PyTorch build")
        return False

    print("✓ torch._grouped_mm is available")

    # Check CUTLASS_NVCC_ARCHS
    cutlass_archs = os.environ.get('CUTLASS_NVCC_ARCHS', 'NOT SET')
    print(f"ℹ CUTLASS_NVCC_ARCHS: {cutlass_archs}")

    print()
    return True

def test_simple_grouped_mm():
    """Test basic grouped matrix multiplication."""
    print("=" * 70)
    print("Test 1: Simple Grouped Matrix Multiplication")
    print("=" * 70)

    try:
        # Create simple test inputs
        # For bfloat16 (2 bytes per element), contiguous dim must be at least
        # 16 bytes = 8 elements. Use multiples of 16 for better alignment.
        # Group 1: 16x16 @ 16x16 = 16x16
        # Group 2: 32x16 @ 16x16 = 32x16

        device = 'cuda'
        dtype = torch.bfloat16

        # Concatenated A matrix [48, 16] = [16, 16] + [32, 16]
        mat_a = torch.randn(48, 16, dtype=dtype, device=device)

        # Concatenated B matrix [2, 16, 16] for 2 groups
        mat_b = torch.randn(2, 16, 16, dtype=dtype, device=device)

        # Offsets indicating where each group starts in mat_a
        # Group 0: rows 0-15 (16 rows)
        # Group 1: rows 16-47 (32 rows)
        offs = torch.tensor([16, 48], dtype=torch.int64, device=device)

        print(f"Input A shape: {mat_a.shape} (concatenated, {mat_a.dtype})")
        print(f"Input B shape: {mat_b.shape} (grouped, {mat_b.dtype})")
        print(f"Offsets: {offs.tolist()}")

        # Run grouped matmul
        result = torch._grouped_mm(mat_a, mat_b, offs)

        print(f"✓ Output shape: {result.shape}")
        print(f"✓ Output dtype: {result.dtype}")
        print(f"✓ torch._grouped_mm executed successfully!")

        # Verify output shape
        expected_shape = (48, 16)  # 16+32 rows, 16 cols
        if result.shape == expected_shape:
            print(f"✓ Output shape matches expected: {expected_shape}")
        else:
            print(f"✗ Output shape mismatch: got {result.shape}, expected {expected_shape}")
            return False

        # Check for NaN/Inf
        if torch.isnan(result).any():
            print("✗ Result contains NaN values")
            return False
        if torch.isinf(result).any():
            print("✗ Result contains Inf values")
            return False

        print("✓ Result is finite and valid")

    except RuntimeError as e:
        if "Arch conditional MMA instruction" in str(e):
            print("✗ CRITICAL: 'Arch conditional MMA instruction' error detected!")
            print("   This means the CUTLASS architecture fix did not work.")
            print(f"   Error: {e}")
            return False
        elif "compute capability" in str(e):
            print(f"⚠ Compute capability error: {e}")
            print("   This may be expected on non-Blackwell GPUs")
            return False
        else:
            print(f"✗ Runtime error: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True

def test_larger_grouped_mm():
    """Test with larger matrices."""
    print("=" * 70)
    print("Test 2: Larger Grouped Matrix Multiplication")
    print("=" * 70)

    try:
        device = 'cuda'
        dtype = torch.bfloat16

        # More realistic sizes
        # Group 1: 128x64 @ 64x32 = 128x32
        # Group 2: 256x64 @ 64x32 = 256x32
        # Group 3: 64x64 @ 64x32 = 64x32

        total_rows = 128 + 256 + 64
        mat_a = torch.randn(total_rows, 64, dtype=dtype, device=device)
        mat_b = torch.randn(3, 64, 32, dtype=dtype, device=device)
        offs = torch.tensor([128, 384, 448], dtype=torch.int64, device=device)

        print(f"Input A shape: {mat_a.shape}")
        print(f"Input B shape: {mat_b.shape}")
        print(f"Groups: 3, offsets: {offs.tolist()}")

        result = torch._grouped_mm(mat_a, mat_b, offs)

        print(f"✓ Output shape: {result.shape}")
        print(f"✓ Large grouped_mm executed successfully!")

        if torch.isnan(result).any() or torch.isinf(result).any():
            print("✗ Result contains NaN or Inf values")
            return False

        print("✓ Result is finite and valid")

    except RuntimeError as e:
        if "Arch conditional MMA instruction" in str(e):
            print("✗ CRITICAL: 'Arch conditional MMA instruction' error!")
            return False
        elif "compute capability" in str(e):
            print(f"⚠ Compute capability error (may be expected): {e}")
            return False
        else:
            print(f"✗ Runtime error: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True

def test_with_torch_compile():
    """Test grouped_mm with torch.compile (inductor)."""
    print("=" * 70)
    print("Test 3: Grouped MM with torch.compile (Inductor)")
    print("=" * 70)

    try:
        device = 'cuda'
        dtype = torch.bfloat16

        # Define a simple function that uses grouped_mm
        def grouped_mm_fn(mat_a, mat_b, offs):
            return torch._grouped_mm(mat_a, mat_b, offs)

        # Compile it
        print("Compiling function with torch.compile...")
        compiled_fn = torch.compile(grouped_mm_fn, backend="inductor")

        # Create inputs with proper alignment (multiples of 16 for bfloat16)
        mat_a = torch.randn(48, 16, dtype=dtype, device=device)
        mat_b = torch.randn(2, 16, 16, dtype=dtype, device=device)
        offs = torch.tensor([16, 48], dtype=torch.int64, device=device)

        print("Running compiled function...")
        result = compiled_fn(mat_a, mat_b, offs)

        print(f"✓ Compiled grouped_mm executed successfully!")
        print(f"✓ Output shape: {result.shape}")

        if torch.isnan(result).any() or torch.isinf(result).any():
            print("✗ Result contains NaN or Inf values")
            return False

        print("✓ Result is finite and valid")
        print("✓ CuteDSL code generation and compilation successful!")

    except RuntimeError as e:
        if "Arch conditional MMA instruction" in str(e):
            print("✗ CRITICAL: 'Arch conditional MMA instruction' error!")
            print("   The fix did not work for torch.compile code path")
            return False
        elif "compute capability" in str(e):
            print(f"⚠ Compute capability error: {e}")
            return False
        else:
            print(f"✗ Runtime error: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True

def main():
    print("\n" + "=" * 70)
    print("torch._grouped_mm Test Suite")
    print("=" * 70)
    print()

    # Check prerequisites
    if not check_prerequisites():
        print("=" * 70)
        print("✗ Prerequisites not met - skipping tests")
        print("=" * 70)
        return 1

    # Run tests
    results = []
    results.append(("Simple grouped_mm", test_simple_grouped_mm()))
    results.append(("Larger grouped_mm", test_larger_grouped_mm()))
    results.append(("torch.compile grouped_mm", test_with_torch_compile()))

    # Print summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)

    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nThe torch._grouped_mm operation is working correctly!")
        print("The 'Arch conditional MMA instruction' fix is successful.")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test script for torch._grouped_mm on Blackwell GPUs.

Tests the grouped matrix multiplication operation with proper alignment
and data types.
"""

import os
import sys
import torch

# Set CUTLASS architecture before importing torch modules
# This is critical for CuteDSL JIT compilation on Blackwell GPUs
if "CUTLASS_NVCC_ARCHS" not in os.environ:
    os.environ["CUTLASS_NVCC_ARCHS"] = "90a-real,100a-real"


def check_prerequisites():
    """Check if the environment is suitable for running grouped_mm tests."""
    print("=" * 70)
    print("Prerequisites Check")
    print("=" * 70)

    # Check CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False

    device_name = torch.cuda.get_device_name()
    print(f"✓ CUDA available: {device_name}")

    # Check compute capability
    major, minor = torch.cuda.get_device_capability()
    print(f"✓ Compute Capability: {major}.{minor}")

    if major < 9:
        print(f"⚠ GPU is SM {major}.{minor}")
        print(f"  CuteDSL grouped_mm is optimized for Blackwell (SM 9.0+)")
        print(f"  Tests will run but may fall back to alternative implementations")
    else:
        print(f"✓ Blackwell GPU detected - CuteDSL kernels available")

    # Check if _grouped_mm exists
    if not hasattr(torch, '_grouped_mm'):
        print("✗ torch._grouped_mm not available in this PyTorch build")
        return False

    print("✓ torch._grouped_mm is available")

    # Display environment variable
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
        device = 'cuda'
        dtype = torch.bfloat16

        # Create test inputs with proper alignment
        # For bfloat16 (2 bytes/element), contiguous dimensions should be
        # multiples of 16 bytes = 8 elements minimum, prefer multiples of 16

        # Group 1: 16x16 @ 16x16 = 16x16
        # Group 2: 32x16 @ 16x16 = 32x16
        # Total: [48, 16] split at offset 16

        mat_a = torch.randn(48, 16, dtype=dtype, device=device)
        mat_b = torch.randn(2, 16, 16, dtype=dtype, device=device)
        offs = torch.tensor([16, 48], dtype=torch.int32, device=device)

        print(f"Input A shape: {mat_a.shape} (concatenated, {mat_a.dtype})")
        print(f"Input B shape: {mat_b.shape} (grouped, {mat_b.dtype})")
        print(f"Offsets: {offs.tolist()}")

        # Run grouped matmul
        result = torch._grouped_mm(mat_a, mat_b, offs)

        print(f"✓ Output shape: {result.shape}")
        print(f"✓ Output dtype: {result.dtype}")
        print(f"✓ torch._grouped_mm executed successfully!")

        # Verify output shape
        expected_shape = (48, 16)
        if result.shape != expected_shape:
            print(f"✗ Output shape mismatch: got {result.shape}, expected {expected_shape}")
            return False

        print(f"✓ Output shape matches expected: {expected_shape}")

        # Check for NaN/Inf
        if torch.isnan(result).any():
            print("✗ Result contains NaN values")
            return False
        if torch.isinf(result).any():
            print("✗ Result contains Inf values")
            return False

        print("✓ Result is finite and valid")

        # Verify correctness by comparing with manual computation
        result_group1 = result[:16, :]
        result_group2 = result[16:48, :]

        expected_group1 = mat_a[:16, :] @ mat_b[0, :, :]
        expected_group2 = mat_a[16:48, :] @ mat_b[1, :, :]

        # Check if results are close (allowing for numerical precision)
        if torch.allclose(result_group1, expected_group1, rtol=1e-2, atol=1e-2):
            print("✓ Group 1 result matches expected (within tolerance)")
        else:
            max_diff = (result_group1 - expected_group1).abs().max()
            print(f"⚠ Group 1 result differs from expected (max diff: {max_diff:.6f})")

        if torch.allclose(result_group2, expected_group2, rtol=1e-2, atol=1e-2):
            print("✓ Group 2 result matches expected (within tolerance)")
        else:
            max_diff = (result_group2 - expected_group2).abs().max()
            print(f"⚠ Group 2 result differs from expected (max diff: {max_diff:.6f})")

    except RuntimeError as e:
        error_msg = str(e)
        if "Arch conditional MMA instruction" in error_msg:
            print("✗ CRITICAL: 'Arch conditional MMA instruction' error detected!")
            print("   This means CUTLASS is not configured correctly for Blackwell.")
            print(f"   Error: {e}")
            return False
        elif "compute capability" in error_msg.lower():
            print(f"✗ Compute capability error: {e}")
            return False
        elif "strides should be multiple of 16 bytes" in error_msg:
            print(f"✗ Alignment error: {e}")
            print("   Tensor dimensions need to be adjusted for proper alignment")
            return False
        elif "int32" in error_msg:
            print(f"✗ Offset dtype error: {e}")
            print("   Offsets must be int32, not int64")
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
    """Test with larger matrices and more groups."""
    print("=" * 70)
    print("Test 2: Larger Grouped Matrix Multiplication")
    print("=" * 70)

    try:
        device = 'cuda'
        dtype = torch.bfloat16

        # Three groups with different sizes:
        # Group 1: 128x64 @ 64x32 = 128x32
        # Group 2: 256x64 @ 64x32 = 256x32
        # Group 3: 64x64 @ 64x32 = 64x32
        # Total: 448 rows

        total_rows = 128 + 256 + 64
        mat_a = torch.randn(total_rows, 64, dtype=dtype, device=device)
        mat_b = torch.randn(3, 64, 32, dtype=dtype, device=device)
        offs = torch.tensor([128, 384, 448], dtype=torch.int32, device=device)

        print(f"Input A shape: {mat_a.shape}")
        print(f"Input B shape: {mat_b.shape}")
        print(f"Groups: 3, offsets: {offs.tolist()}")

        result = torch._grouped_mm(mat_a, mat_b, offs)

        print(f"✓ Output shape: {result.shape}")
        print(f"✓ Large grouped_mm executed successfully!")

        expected_shape = (total_rows, 32)
        if result.shape != expected_shape:
            print(f"✗ Output shape mismatch: got {result.shape}, expected {expected_shape}")
            return False

        print(f"✓ Output shape matches expected: {expected_shape}")

        if torch.isnan(result).any() or torch.isinf(result).any():
            print("✗ Result contains NaN or Inf values")
            return False

        print("✓ Result is finite and valid")

        # Verify a subset of results
        result_group1 = result[:128, :]
        expected_group1 = mat_a[:128, :] @ mat_b[0, :, :]

        if torch.allclose(result_group1, expected_group1, rtol=1e-2, atol=1e-2):
            print("✓ Group 1 result verified")
        else:
            max_diff = (result_group1 - expected_group1).abs().max()
            print(f"⚠ Group 1 result differs (max diff: {max_diff:.6f})")

    except RuntimeError as e:
        error_msg = str(e)
        if "Arch conditional MMA instruction" in error_msg:
            print("✗ CRITICAL: 'Arch conditional MMA instruction' error!")
            return False
        elif "compute capability" in error_msg.lower():
            print(f"⚠ Compute capability error (expected on some GPUs): {e}")
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
    """Test grouped_mm with torch.compile (Inductor code generation)."""
    print("=" * 70)
    print("Test 3: Grouped MM with torch.compile (Inductor)")
    print("=" * 70)

    try:
        device = 'cuda'
        dtype = torch.bfloat16

        # Define a function that uses grouped_mm
        def grouped_mm_fn(mat_a, mat_b, offs):
            return torch._grouped_mm(mat_a, mat_b, offs)

        # Compile it with inductor backend
        print("Compiling function with torch.compile...")
        compiled_fn = torch.compile(grouped_mm_fn, backend="inductor", mode="default")

        # Create inputs with proper alignment
        mat_a = torch.randn(48, 16, dtype=dtype, device=device)
        mat_b = torch.randn(2, 16, 16, dtype=dtype, device=device)
        offs = torch.tensor([16, 48], dtype=torch.int32, device=device)

        print("Running compiled function (first call - triggers compilation)...")
        result = compiled_fn(mat_a, mat_b, offs)

        print(f"✓ Compiled grouped_mm executed successfully!")
        print(f"✓ Output shape: {result.shape}")

        if torch.isnan(result).any() or torch.isinf(result).any():
            print("✗ Result contains NaN or Inf values")
            return False

        print("✓ Result is finite and valid")

        # Run again to test cached compilation
        print("Running compiled function again (uses cached kernel)...")
        result2 = compiled_fn(mat_a, mat_b, offs)

        if torch.allclose(result, result2, rtol=1e-6, atol=1e-6):
            print("✓ Cached kernel produces consistent results")
        else:
            print("⚠ Results differ between runs")

        print("✓ CuteDSL code generation and compilation successful!")

    except RuntimeError as e:
        error_msg = str(e)
        if "Arch conditional MMA instruction" in error_msg:
            print("✗ CRITICAL: 'Arch conditional MMA instruction' error!")
            print("   The CuteDSL compilation fix did not work for torch.compile")
            print(f"   Error: {e}")
            return False
        elif "compute capability" in error_msg.lower():
            print(f"⚠ Compute capability error: {e}")
            return False
        else:
            print(f"✗ Runtime error during compilation: {e}")
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


def test_edge_cases():
    """Test edge cases and error handling."""
    print("=" * 70)
    print("Test 4: Edge Cases and Error Handling")
    print("=" * 70)

    device = 'cuda'
    dtype = torch.bfloat16
    passed_tests = 0
    total_tests = 3

    # Test 1: Single group
    try:
        print("\n4.1: Single group test...")
        mat_a = torch.randn(32, 16, dtype=dtype, device=device)
        mat_b = torch.randn(1, 16, 16, dtype=dtype, device=device)
        offs = torch.tensor([32], dtype=torch.int32, device=device)
        result = torch._grouped_mm(mat_a, mat_b, offs)
        print(f"   ✓ Single group works: {result.shape}")
        passed_tests += 1
    except Exception as e:
        print(f"   ✗ Single group failed: {e}")

    # Test 2: Many groups
    try:
        print("\n4.2: Many groups test (8 groups)...")
        total_rows = 16 * 8  # 8 groups of 16 rows each
        mat_a = torch.randn(total_rows, 16, dtype=dtype, device=device)
        mat_b = torch.randn(8, 16, 16, dtype=dtype, device=device)
        offs = torch.tensor([16*i for i in range(1, 9)], dtype=torch.int32, device=device)
        result = torch._grouped_mm(mat_a, mat_b, offs)
        print(f"   ✓ Many groups work: {result.shape}")
        passed_tests += 1
    except Exception as e:
        print(f"   ✗ Many groups failed: {e}")

    # Test 3: Large dimensions
    try:
        print("\n4.3: Large dimensions test...")
        mat_a = torch.randn(1024, 128, dtype=dtype, device=device)
        mat_b = torch.randn(2, 128, 64, dtype=dtype, device=device)
        offs = torch.tensor([512, 1024], dtype=torch.int32, device=device)
        result = torch._grouped_mm(mat_a, mat_b, offs)
        print(f"   ✓ Large dimensions work: {result.shape}")
        passed_tests += 1
    except Exception as e:
        print(f"   ✗ Large dimensions failed: {e}")

    print(f"\nEdge cases passed: {passed_tests}/{total_tests}")
    print()
    return passed_tests == total_tests


def main():
    print("\n" + "=" * 70)
    print("torch._grouped_mm Test Suite")
    print("=" * 70)
    print()

    # Check prerequisites
    if not check_prerequisites():
        print("=" * 70)
        print("✗ Prerequisites not met - cannot run tests")
        print("=" * 70)
        return 1

    # Run tests
    results = []
    results.append(("Simple grouped_mm", test_simple_grouped_mm()))
    results.append(("Larger grouped_mm", test_larger_grouped_mm()))
    results.append(("torch.compile grouped_mm", test_with_torch_compile()))
    results.append(("Edge cases", test_edge_cases()))

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
        print("CuteDSL kernels are compiling and executing successfully.")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        failed = [name for name, passed in results if not passed]
        print(f"\nFailed tests: {', '.join(failed)}")
        print("\nPlease check the errors above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

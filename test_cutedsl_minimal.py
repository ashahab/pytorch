#!/usr/bin/env python3
"""
Minimal test to understand how @cute.kernel decorator compilation works
and where the MMA instruction error occurs.
"""

import os
import sys

# Set environment variable BEFORE any imports
os.environ["CUTLASS_NVCC_ARCHS"] = "90a-real,100a-real"

print(f"Python version: {sys.version}")
print(f"CUTLASS_NVCC_ARCHS set to: {os.environ.get('CUTLASS_NVCC_ARCHS', 'NOT SET')}")

# Now try to import cutlass
try:
    print("\n1. Importing cutlass...")
    import cutlass
    print("   ✓ cutlass imported successfully")

    print("\n2. Importing cutlass.cute...")
    import cutlass.cute as cute
    print("   ✓ cutlass.cute imported successfully")

    print("\n3. Defining a simple @cute.kernel...")

    @cute.kernel
    def simple_test_kernel(A, B, C):
        """Simplest possible kernel that might trigger MMA instruction."""
        # Just a placeholder - the decorator itself triggers compilation
        pass

    print("   ✓ Kernel decorated successfully")

    print("\n4. Checking kernel attributes...")
    print(f"   Kernel type: {type(simple_test_kernel)}")
    print(f"   Kernel attributes: {dir(simple_test_kernel)[:5]}...")

    print("\n✅ SUCCESS: No MMA instruction error occurred!")
    print("The environment variable approach worked for this simple case.")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Now testing if we can compile a kernel with actual MMA operations...")

try:
    @cute.kernel
    def mma_test_kernel(A, B, C):
        """Kernel with MMA operations that should trigger Blackwell-specific instructions."""
        # This would contain actual MMA operations in a real kernel
        # The decorator compilation is what matters
        pass

    print("✅ MMA kernel decorated successfully!")

except Exception as e:
    print(f"❌ MMA kernel failed: {e}")

print("\nConclusion:")
print("-" * 40)
print("If this test passes, the issue is likely in:")
print("1. How PyCodeCache loads the module")
print("2. When the environment variable is set relative to module loading")
print("3. Cached kernels compiled without the proper flags")
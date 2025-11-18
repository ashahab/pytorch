#!/usr/bin/env python3
"""
Simple test to verify CUTLASS_NVCC_ARCHS is set correctly in generated code.

This test doesn't require nvidia-cutlass to be installed.
It just checks that the environment variable setting code exists in the right places.
"""

import os
import re
from pathlib import Path

def check_file_has_env_var_setting(filepath, description):
    """Check if a file sets CUTLASS_NVCC_ARCHS before importing cutlass."""
    print(f"\nChecking: {description}")
    print(f"  File: {filepath}")

    if not Path(filepath).exists():
        print(f"  ✗ File not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if file has env var setting
    has_env_var = 'CUTLASS_NVCC_ARCHS' in content and '90a-real,100a-real' in content

    # Check if file imports cutlass
    has_cutlass_import = re.search(r'^\s*(import cutlass|from cutlass)', content, re.MULTILINE)

    if not has_cutlass_import:
        print(f"  ℹ No cutlass imports found (OK)")
        return True

    if not has_env_var:
        print(f"  ✗ MISSING: CUTLASS_NVCC_ARCHS setting not found")
        return False

    # Find positions
    env_var_pos = content.find('CUTLASS_NVCC_ARCHS')
    cutlass_import_match = has_cutlass_import
    cutlass_import_pos = cutlass_import_match.start()

    if env_var_pos < cutlass_import_pos:
        print(f"  ✓ CUTLASS_NVCC_ARCHS set BEFORE cutlass import")
        return True
    else:
        print(f"  ✗ WRONG ORDER: CUTLASS_NVCC_ARCHS set AFTER cutlass import")
        print(f"     Env var position: {env_var_pos}")
        print(f"     Import position: {cutlass_import_pos}")
        return False

def main():
    print("=" * 70)
    print("CuteDSL Environment Variable Configuration Validation")
    print("=" * 70)

    pytorch_root = Path(__file__).parent

    files_to_check = [
        (
            pytorch_root / "torch/_inductor/codegen/cutedsl/_cutedsl_utils.py",
            "CuteDSL Utils Module"
        ),
        (
            pytorch_root / "torch/_inductor/kernel/vendored_templates/cutedsl_grouped_gemm.py",
            "Vendored CuteDSL Grouped GEMM Template"
        ),
        (
            pytorch_root / "torch/_inductor/kernel/templates/cutedsl_mm_grouped.py.jinja",
            "CuteDSL Grouped MM Jinja Template"
        ),
    ]

    all_passed = True
    for filepath, description in files_to_check:
        if not check_file_has_env_var_setting(filepath, description):
            all_passed = False

    # Check generated code from gen_imports
    print(f"\nChecking: Generated Code from cutedsl_kernel.py gen_imports()")
    kernel_py = pytorch_root / "torch/_inductor/codegen/cutedsl/cutedsl_kernel.py"
    if kernel_py.exists():
        with open(kernel_py, 'r') as f:
            content = f.read()

        # Look for gen_imports method
        gen_imports_match = re.search(r'def gen_imports\(.*?\):(.*?)^\s{4}def\s', content, re.DOTALL | re.MULTILINE)
        if gen_imports_match:
            gen_imports_code = gen_imports_match.group(1)
            has_env_var = 'CUTLASS_NVCC_ARCHS' in gen_imports_code
            has_cutlass = 'import cutlass' in gen_imports_code

            if has_cutlass:
                if has_env_var:
                    # Check order in the generated string
                    env_pos = gen_imports_code.find('CUTLASS_NVCC_ARCHS')
                    cutlass_pos = gen_imports_code.find('import cutlass')
                    if env_pos < cutlass_pos:
                        print(f"  ✓ gen_imports() sets CUTLASS_NVCC_ARCHS BEFORE cutlass import")
                    else:
                        print(f"  ✗ gen_imports() sets CUTLASS_NVCC_ARCHS AFTER cutlass import")
                        all_passed = False
                else:
                    print(f"  ✗ gen_imports() missing CUTLASS_NVCC_ARCHS setting")
                    all_passed = False
            else:
                print(f"  ℹ gen_imports() doesn't import cutlass (unusual)")
        else:
            print(f"  ⚠ Couldn't parse gen_imports() method")
    else:
        print(f"  ✗ File not found: {kernel_py}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("=" * 70)
        print("\nThe CUTLASS_NVCC_ARCHS environment variable is correctly set")
        print("BEFORE all cutlass imports. This should fix the 'Arch conditional")
        print("MMA instruction' error.")
        print("\nNext step: Test in your actual training environment where")
        print("nvidia-cutlass is installed and CuteDSL kernels are compiled.")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 70)
        print("\nThe environment variable configuration is not correct.")
        print("Please review the failed checks above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

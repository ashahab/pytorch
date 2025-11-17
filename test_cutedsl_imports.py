#!/usr/bin/env python3
"""
Comprehensive test of all CuteDSL imports in PyTorch codebase.
This script verifies that all import statements used in the code are valid.
"""

import sys
import os

# Test results
passed = []
failed = []

def test_import(description, import_statement):
    """Test an import statement"""
    print(f"Testing: {description}")
    print(f"  {import_statement}")
    try:
        exec(import_statement)
        print("  ✓ PASS")
        passed.append(description)
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        failed.append((description, str(e)))
        return False

print("=" * 80)
print("CUTEDSL IMPORT VERIFICATION TEST")
print("=" * 80)
print()

# Test 1: Main imports from mm_scaled_grouped.py
test_import(
    "CuteDSLTemplate from cutedsl_template",
    "from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate"
)
print()

test_import(
    "get_groupgemm_configs from template_heuristics.cutedsl",
    "from torch._inductor.template_heuristics.cutedsl import get_groupgemm_configs"
)
print()

# Test 2: Imports from cutedsl_template.py
test_import(
    "CuteDSLBenchmarkRequest from autotune_process",
    "from torch._inductor.autotune_process import CuteDSLBenchmarkRequest"
)
print()

test_import(
    "CuteDSLTemplateBuffer from ir",
    "from torch._inductor.ir import CuteDSLTemplateBuffer"
)
print()

test_import(
    "CuteDSLTemplateKernel from cutedsl_kernel",
    "from torch._inductor.codegen.cutedsl.cutedsl_kernel import CuteDSLTemplateKernel"
)
print()

# Test 3: Imports from cutedsl __init__.py
test_import(
    "CuteDSLTemplate from cutedsl package",
    "from torch._inductor.codegen.cutedsl import CuteDSLTemplate"
)
print()

test_import(
    "CuteDSLTemplateCaller from cutedsl package",
    "from torch._inductor.codegen.cutedsl import CuteDSLTemplateCaller"
)
print()

# Test 4: Module-level imports
test_import(
    "cutedsl module",
    "import torch._inductor.template_heuristics.cutedsl"
)
print()

test_import(
    "cutedsl codegen package",
    "import torch._inductor.codegen.cutedsl"
)
print()

# Test 5: Verify helper functions exist
test_import(
    "load_template from utils",
    "from torch._inductor.utils import load_template"
)
print()

test_import(
    "use_blackwell_cutedsl_grouped_mm from utils",
    "from torch._inductor.utils import use_blackwell_cutedsl_grouped_mm"
)
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Passed: {len(passed)}")
print(f"Failed: {len(failed)}")
print()

if failed:
    print("Failed imports:")
    for desc, error in failed:
        print(f"  ✗ {desc}")
        print(f"    Error: {error}")
    sys.exit(1)
else:
    print("✓ All imports successful!")
    sys.exit(0)

# Cherry-Pick Plan: Blackwell CuTeDSL Support for `_grouped_mm`

## Executive Summary

Cherry-pick commit `07fbf93fe555fe105a3a1957263ca8b43b4a14d7` to add Blackwell GPU support for grouped matrix multiplication using NVIDIA's CuTeDSL (CuTe Domain Specific Language) kernels.

**Target Branch:** `abin-shahab/v2.8.0`
**Source Commit:** `07fbf93fe555fe105a3a1957263ca8b43b4a14d7`
**Author:** Nikhil Patel <nikhilap@meta.com>
**Original PR:** #167340

---

## Commit Overview

### What This Adds

This commit introduces a high-performance Blackwell-specific implementation of grouped GEMM (General Matrix Multiply) operations using NVIDIA's CuTeDSL framework. Key features:

1. **New CuTeDSL Template:** Jinja2 template for generating Blackwell-optimized grouped GEMM kernels
2. **Heuristics System:** Autotuning configurations for different problem sizes
3. **Build System Integration:** Automatic vendoring of CUTLASS grouped_gemm.py during build
4. **Comprehensive Testing:** New test suite with 150+ lines covering various layouts and sizes
5. **GPU Detection:** Proper Blackwell architecture detection to prevent activation on non-Blackwell GPUs

### Files Modified (10 files, +833/-31 lines)

1. `.ci/pytorch/test.sh` - Add CuTeDSL tests to CI smoke tests
2. `.gitignore` - Ignore vendored templates directory
3. `setup.py` - Add `mirror_inductor_external_kernels()` function to copy CUTLASS files
4. `test/inductor/test_cutedsl_grouped_mm.py` - **New file:** comprehensive test suite
5. `torch/_inductor/config.py` - Add `cutedsl_enable_autotuning` config flag
6. `torch/_inductor/kernel/mm_common.py` - Add `load_kernel_template()` helper
7. `torch/_inductor/kernel/mm_grouped.py` - Integrate CuTeDSL template into grouped MM
8. `torch/_inductor/kernel/templates/cutedsl_mm_grouped.py.jinja` - **New file:** kernel template
9. `torch/_inductor/template_heuristics/cutedsl.py` - **New file:** autotuning heuristics
10. `torch/_inductor/utils.py` - Add utilities for CuTeDSL availability checks

---

## Prerequisites

### Required Dependencies

1. **NVIDIA Blackwell GPU (Compute Capability 10.x)**
   - B200, B200A, or equivalent
   - Non-Blackwell GPUs will gracefully fall back to existing implementations

2. **CUDA Toolkit**
   - CUDA 12.8+ recommended (tested in original commit)
   - CUDA 12.6+ minimum

3. **NVIDIA CuTeDSL Library**
   - `nvidia-cutlass-dsl` version 4.2.1+
   - Available via pip: `pip install nvidia-cutlass-dsl>=4.2.1`

4. **CUTLASS Submodule**
   - File must exist: `third_party/cutlass/examples/python/CuTeDSL/blackwell/grouped_gemm.py`
   - Verify: `ls third_party/cutlass/examples/python/CuTeDSL/blackwell/grouped_gemm.py`

### Build Environment

```bash
# Recommended environment variables for testing
export CUDA_VISIBLE_DEVICES=0  # Select Blackwell GPU
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
export CUTEDSL_ENABLE_AUTOTUNING=1  # Enable autotuning
```

---

## Step-by-Step Cherry-Pick Plan

### Phase 1: Preparation (15 minutes)

#### 1.1 Verify Prerequisites
```bash
# Check current branch
git branch --show-current  # Should be: abin-shahab/v2.8.0

# Verify CUTLASS submodule has required file
ls -l third_party/cutlass/examples/python/CuTeDSL/blackwell/grouped_gemm.py

# Check if commit is already applied
git log --oneline --grep="Blackwell CuTeDSL" | head -1

# Verify clean working directory
git status --porcelain
```

#### 1.2 Create Backup Branch
```bash
git branch backup/pre-cutedsl-cherrypick-$(date +%Y%m%d)
```

#### 1.3 Check for Potential Conflicts
```bash
# Check if any of the 10 files have been modified since merge-base
git diff HEAD 07fbf93fe555fe105a3a1957263ca8b43b4a14d7 --stat -- \
  .ci/pytorch/test.sh \
  .gitignore \
  setup.py \
  torch/_inductor/config.py \
  torch/_inductor/kernel/mm_common.py \
  torch/_inductor/kernel/mm_grouped.py \
  torch/_inductor/utils.py
```

---

### Phase 2: Cherry-Pick Execution (10 minutes)

#### 2.1 Perform Cherry-Pick
```bash
# Cherry-pick the commit
git cherry-pick 07fbf93fe555fe105a3a1957263ca8b43b4a14d7

# Expected outcome: Clean cherry-pick or manageable conflicts
```

#### 2.2 Resolve Conflicts (if any)

**Likely Conflict Areas:**

1. **`setup.py`**
   - **Cause:** Other cherry-picks may have modified build logic
   - **Resolution:**
     - Keep the `mirror_inductor_external_kernels()` function
     - Ensure it's called after `build_deps()` in `main()`
     - Merge package_data entries for `_inductor/kernel/templates/*.jinja`

2. **`.gitignore`**
   - **Cause:** Other entries may have been added
   - **Resolution:** Simply add `torch/_inductor/kernel/vendored_templates/*` to list

3. **`torch/_inductor/kernel/mm_grouped.py`**
   - **Cause:** Other optimizations or backends may have been added
   - **Resolution:** Carefully merge the CuTeDSL integration logic

**Conflict Resolution Strategy:**
```bash
# If conflicts occur:
git status  # Identify conflicted files

# For each conflicted file:
# 1. Open in editor
# 2. Search for "<<<<<<< HEAD" markers
# 3. Resolve by keeping both changes where appropriate
# 4. Test that syntax is valid: python -m py_compile <file>

# After resolving all conflicts:
git add <resolved-files>
git cherry-pick --continue
```

#### 2.3 Verify Cherry-Pick Success
```bash
# Check commit was applied
git log -1 --oneline

# Verify all 10 files were modified
git show HEAD --stat

# Ensure new files were created
ls test/inductor/test_cutedsl_grouped_mm.py
ls torch/_inductor/kernel/templates/cutedsl_mm_grouped.py.jinja
ls torch/_inductor/template_heuristics/cutedsl.py
```

---

### Phase 3: Compilation Testing (30-45 minutes)

#### 3.1 Clean Build Environment
```bash
# Remove stale build artifacts
python setup.py clean
rm -rf build/

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
```

#### 3.2 Build PyTorch with CuTeDSL Support

**Option A: Full Debug Build (Recommended for Testing)**
```bash
# Set build flags
export DEBUG=1
export USE_CUDA=1
export MAX_JOBS=16  # Adjust based on available cores

# Build in development mode
python setup.py develop

# Expected duration: 20-30 minutes (full build)
# Expected duration: 5-10 minutes (incremental if minimal conflicts)
```

**Option B: Optimized Build**
```bash
export REL_WITH_DEB_INFO=1
export USE_CUDA=1
export MAX_JOBS=16

python setup.py develop
```

#### 3.3 Verify Build Success

**Check for vendored files:**
```bash
# Verify setup.py copied CUTLASS file
ls -l torch/_inductor/kernel/vendored_templates/cutedsl_grouped_gemm.py

# Verify Jinja template exists
ls -l torch/_inductor/kernel/templates/cutedsl_mm_grouped.py.jinja

# Check Python module imports
python -c "from torch._inductor.kernel.vendored_templates.cutedsl_grouped_gemm import GroupedGemmKernel; print('Import successful')"
```

**Check for compilation errors:**
```bash
# Test basic PyTorch functionality
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Test inductor imports
python -c "from torch._inductor.template_heuristics.cutedsl import get_groupgemm_configs; print('CuTeDSL heuristics loaded')"
python -c "from torch._inductor.utils import ensure_cute_available; print(f'CuTeDSL available: {ensure_cute_available()}')"
```

**Check GPU architecture detection:**
```bash
python -c "
from torch._inductor.codegen.cuda.cuda_env import is_datacenter_blackwell_arch
print(f'Blackwell GPU detected: {is_datacenter_blackwell_arch()}')
"
```

#### 3.4 Build Troubleshooting

**Common Issues and Solutions:**

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Missing CUTLASS file** | `FileNotFoundError: third_party/cutlass/...` | Update submodule: `git submodule update --init --recursive third_party/cutlass` |
| **Import error for vendored file** | `ModuleNotFoundError: torch._inductor.kernel.vendored_templates` | Re-run `python setup.py develop` to trigger `mirror_inductor_external_kernels()` |
| **Jinja2 template not found** | `TemplateNotFound: cutedsl_mm_grouped` | Verify file in `torch/_inductor/kernel/templates/` and check `setup.py` package_data |
| **CUDA compilation error** | `nvcc error` | Check CUDA version compatibility (12.6+) and ensure `USE_CUDA=1` |

---

### Phase 4: Functionality Testing (45-60 minutes)

#### 4.1 Unit Tests

**Run CuTeDSL-specific tests:**
```bash
# Full test suite (if Blackwell GPU available)
python test/inductor/test_cutedsl_grouped_mm.py -v

# Expected output:
# test_grouped_gemm_basic (group_size=2, M_hint=256, K=64, N=128) ... ok
# test_grouped_gemm_basic (group_size=2, M_hint=256, K=64, N=256) ... ok
# ... (multiple combinations)
# test_grouped_gemm_assorted_layouts ... ok
# ----------------------------------------------------------------------
# Ran 50+ tests in X.XXs
# OK
```

**Test individual configurations:**
```bash
# Test basic functionality
python test/inductor/test_cutedsl_grouped_mm.py TestCuTeDSLGroupedGemm.test_grouped_gemm_basic

# Test layout variations
python test/inductor/test_cutedsl_grouped_mm.py TestCuTeDSLGroupedGemm.test_grouped_gemm_assorted_layouts
```

**Expected Test Results:**
- ✅ All tests pass on Blackwell GPU
- ⚠️ Tests skip gracefully on non-Blackwell GPU with message: "CuTeDSL library or Blackwell device not available"

#### 4.2 Integration Tests

**Test with torch.compile:**
```python
# Save as test_cutedsl_integration.py
import torch
from torch._inductor import config

def test_grouped_mm_compiled():
    device = "cuda"
    dtype = torch.bfloat16

    G, M, K, N = 8, 256, 128, 256
    A = torch.randn(G * M, K, dtype=dtype, device=device)
    B = torch.randn(G, K, N, dtype=dtype, device=device)
    offsets = torch.tensor([M * (i + 1) for i in range(G)],
                          dtype=torch.int32, device=device)

    def grouped_gemm_fn(A, B, offs):
        return torch._grouped_mm(A, B, offs=offs)

    # Eager execution
    result_eager = grouped_gemm_fn(A, B, offsets)

    # Compile with CuTeDSL backend
    with config.patch({
        "max_autotune": True,
        "max_autotune_gemm_backends": "CUTEDSL",
    }):
        compiled_fn = torch.compile(grouped_gemm_fn, backend="inductor")
        result_compiled = compiled_fn(A, B, offsets)

    # Verify results match
    torch.testing.assert_close(result_eager, result_compiled, rtol=1e-2, atol=1e-3)
    print("✅ Integration test passed!")

if __name__ == "__main__":
    test_grouped_mm_compiled()
```

```bash
python test_cutedsl_integration.py
```

#### 4.3 Performance Validation

**Benchmark script:**
```python
# Save as benchmark_cutedsl.py
import torch
from torch._inductor import config
import time

def benchmark_grouped_mm(G=16, M=512, K=256, N=512, warmup=10, iters=100):
    device = "cuda"
    dtype = torch.bfloat16

    A = torch.randn(G * M, K, dtype=dtype, device=device)
    B = torch.randn(G, K, N, dtype=dtype, device=device)
    offsets = torch.tensor([M * (i + 1) for i in range(G)],
                          dtype=torch.int32, device=device)

    def grouped_gemm_fn(A, B, offs):
        return torch._grouped_mm(A, B, offs=offs)

    # Compile with CuTeDSL
    with config.patch({
        "max_autotune": True,
        "max_autotune_gemm_backends": "CUTEDSL",
        "cutedsl_enable_autotuning": True,
    }):
        compiled_fn = torch.compile(grouped_gemm_fn, backend="inductor")

        # Warmup
        for _ in range(warmup):
            _ = compiled_fn(A, B, offsets)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            result = compiled_fn(A, B, offsets)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    # Calculate TFLOPS
    flops_per_iter = 2 * G * M * K * N  # 2 for multiply-add
    tflops = (flops_per_iter * iters / elapsed) / 1e12

    print(f"Problem size: G={G}, M={M}, K={K}, N={N}")
    print(f"Average time: {elapsed/iters*1000:.3f} ms")
    print(f"Throughput: {tflops:.2f} TFLOPS")
    print(f"✅ Performance test completed")

if __name__ == "__main__":
    benchmark_grouped_mm()
```

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_cutedsl.py
```

**Expected Performance Characteristics:**
- Blackwell GPU (B200): Significant speedup over Triton baseline (1.5-2x typical)
- Non-Blackwell GPU: Graceful fallback to existing Triton/ATen implementations

#### 4.4 Test Matrix

| Test Category | Test Case | Expected Result | Priority |
|--------------|-----------|-----------------|----------|
| **Unit Tests** | Basic grouped MM | Pass on Blackwell | P0 |
| | Various group sizes (2, 8, 16) | Pass | P0 |
| | Different matrix sizes | Pass | P0 |
| | Layout variations (contiguous, offset, padded, view) | Pass | P0 |
| | Broadcasting | Pass | P1 |
| **Integration** | torch.compile integration | Correct results vs eager | P0 |
| | Autotuning enabled | Chooses best config | P1 |
| | Multiple backends (CUTEDSL + Triton) | CuTeDSL selected on Blackwell | P1 |
| **Regression** | Existing grouped MM tests | No degradation | P0 |
| | Non-Blackwell GPUs | Proper fallback | P0 |
| **Performance** | Blackwell GPU throughput | > baseline TFLOPS | P1 |
| | Memory usage | Within expected bounds | P2 |

#### 4.5 Regression Testing

**Ensure existing functionality still works:**
```bash
# Run broader inductor test suite
python test/inductor/test_max_autotune.py -k grouped -v

# Test that non-CuTeDSL backends still work
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON python test/inductor/test_cutedsl_grouped_mm.py

# Verify fallback on non-Blackwell (if available)
# Should skip gracefully
python test/inductor/test_cutedsl_grouped_mm.py
```

---

### Phase 5: CI Integration (15 minutes)

#### 5.1 Verify CI Changes

The commit modifies `.ci/pytorch/test.sh` to include CuTeDSL tests in smoke tests:

```bash
# Review changes
git diff HEAD~1 .ci/pytorch/test.sh

# Verify new test function exists
grep -A5 "test_python_smoke_b200" .ci/pytorch/test.sh
```

#### 5.2 Local CI Simulation (Optional)

```bash
# Simulate CI test run (if Blackwell GPU available)
export PYTHON_TEST_EXTRA_OPTION=""
python test/run_test.py --include \
  test_matmul_cuda \
  test_scaled_matmul_cuda \
  inductor/test_fp8 \
  inductor/test_max_autotune \
  inductor/test_cutedsl_grouped_mm
```

---

### Phase 6: Documentation and Validation (10 minutes)

#### 6.1 Update Commit Message (if needed)

```bash
# If cherry-pick preserved commit message, verify it:
git log -1 --pretty=format:"%B"

# If you need to amend (e.g., add cherry-pick note):
git commit --amend
# Add: "Cherry-picked to v2.8.0 branch for Blackwell support"
```

#### 6.2 Create Validation Report

```bash
# Save validation results
cat > cutedsl_validation_report.md << 'EOF'
# CuTeDSL Grouped GEMM Validation Report

## Build Status
- ✅ Clean build completed
- ✅ Vendored files copied successfully
- ✅ All imports functional

## Test Results
- ✅ Unit tests: X/X passed
- ✅ Integration tests: All passed
- ✅ Regression tests: No issues detected

## Performance
- Blackwell B200: X.X TFLOPS (baseline: Y.Y TFLOPS, +Z% improvement)

## GPU Compatibility
- ✅ Blackwell GPU: CuTeDSL kernel active
- ✅ Non-Blackwell GPU: Graceful fallback confirmed

## Issues Encountered
- None / [List any issues]

## Sign-off
Validated by: [Your Name]
Date: $(date)
Commit: $(git rev-parse HEAD)
EOF
```

#### 6.3 Final Validation Checklist

- [ ] Cherry-pick applied cleanly or conflicts resolved
- [ ] Build succeeds without errors
- [ ] Vendored file copied: `torch/_inductor/kernel/vendored_templates/cutedsl_grouped_gemm.py`
- [ ] New test file exists: `test/inductor/test_cutedsl_grouped_mm.py`
- [ ] Unit tests pass (or skip gracefully on non-Blackwell)
- [ ] Integration test with torch.compile passes
- [ ] No regression in existing grouped MM functionality
- [ ] Performance validated (if Blackwell GPU available)
- [ ] CI smoke tests include new test
- [ ] Commit message preserved/updated appropriately

---

## Rollback Plan

If issues are discovered post-merge:

### Quick Rollback
```bash
# Revert the cherry-pick commit
git revert HEAD

# Or reset to backup branch
git reset --hard backup/pre-cutedsl-cherrypick-YYYYMMDD
git push --force-with-lease  # If already pushed
```

### Selective Disable
```bash
# Disable CuTeDSL backend via config (temporary)
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON,ATen  # Exclude CUTEDSL
```

---

## Known Issues and Mitigations

### Issue 1: Non-Blackwell GPU False Activation
**Original Bug:** Previous version (#165036) incorrectly activated on non-Blackwell GPUs
**Fix in #167340:** Added proper `is_datacenter_blackwell_arch()` check
**Mitigation:** This cherry-pick includes the fix; test on multiple GPU types

### Issue 2: CuTeDSL Library Not Available
**Symptom:** `ensure_cute_available()` returns False
**Cause:** `nvidia-cutlass-dsl` package not installed
**Mitigation:** Tests gracefully skip; document dependency in release notes

### Issue 3: CUDA Version Mismatch
**Symptom:** Compilation errors with nvcc
**Cause:** CUDA < 12.6
**Mitigation:** Document minimum CUDA version requirement

---

## Success Criteria

### Minimum Requirements (Must Have)
1. ✅ Clean cherry-pick with resolved conflicts (if any)
2. ✅ Successful build with no compilation errors
3. ✅ Unit tests pass on Blackwell GPU OR skip gracefully on non-Blackwell
4. ✅ No regression in existing grouped MM functionality
5. ✅ Vendored files present and importable

### Optimal Requirements (Should Have)
6. ✅ Performance improvement over baseline (1.2x+ on Blackwell)
7. ✅ Integration tests pass
8. ✅ CI smoke tests configured correctly
9. ✅ Documentation updated

### Stretch Goals (Nice to Have)
10. ✅ Autotuning configurations validated for multiple problem sizes
11. ✅ Memory profiling shows no leaks
12. ✅ Tritonbench comparison data collected

---

## Timeline Estimate

| Phase | Duration | Can Parallelize |
|-------|----------|----------------|
| Preparation | 15 min | No |
| Cherry-pick | 10 min | No |
| Compilation | 30-45 min | No |
| Unit Tests | 20 min | Partially |
| Integration Tests | 15 min | Partially |
| Performance Tests | 10 min | Partially |
| CI Integration | 15 min | Yes |
| Documentation | 10 min | Yes |
| **Total** | **~2 hours** | |

**Estimated Total Time:** 2-3 hours (with Blackwell GPU access)
**Without Blackwell GPU:** ~1 hour (build + verify graceful fallback)

---

## Contact and Escalation

- **Original Author:** Nikhil Patel <nikhilap@meta.com>
- **Original Reviewer:** @jananisriram
- **PyTorch Slack:** #inductor channel for questions
- **Escalation:** If critical issues arise, document and create GitHub issue referencing PR #167340

---

## Appendix A: Environment Variables Reference

| Variable | Values | Purpose |
|----------|--------|---------|
| `CUTEDSL_ENABLE_AUTOTUNING` | 0/1 | Enable CuTeDSL autotuning |
| `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM` | 0/1 | Enable max autotune for GEMM |
| `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` | CUTEDSL,TRITON,ATen | Select backends |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES` | 0/1 | Disable compilation caches (testing) |
| `TORCH_LOGS` | +inductor | Enable inductor logging |
| `CUDA_VISIBLE_DEVICES` | GPU ID | Select specific GPU |
| `USE_CUDA` | 0/1 | Enable CUDA in build |
| `DEBUG` | 0/1 | Debug build with symbols |
| `MAX_JOBS` | N | Parallel build jobs |

---

## Appendix B: Quick Command Reference

```bash
# Full cherry-pick workflow
git cherry-pick 07fbf93fe555fe105a3a1957263ca8b43b4a14d7
python setup.py clean && python setup.py develop
python test/inductor/test_cutedsl_grouped_mm.py -v

# Verify build artifacts
ls torch/_inductor/kernel/vendored_templates/cutedsl_grouped_gemm.py
ls torch/_inductor/kernel/templates/cutedsl_mm_grouped.py.jinja

# Check GPU compatibility
python -c "from torch._inductor.codegen.cuda.cuda_env import is_datacenter_blackwell_arch; print(is_datacenter_blackwell_arch())"

# Run with CuTeDSL forced
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=CUTEDSL python test/inductor/test_cutedsl_grouped_mm.py

# Rollback if needed
git revert HEAD
```

---

**Plan Version:** 1.0
**Last Updated:** 2025-11-14
**Status:** Ready for Execution

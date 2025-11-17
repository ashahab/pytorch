# Blackwell CuTeDSL Cherry-Pick: Quick Execution Checklist

## Pre-Flight Checklist
- [ ] On correct branch: `abin-shahab/v2.8.0`
- [ ] Clean working directory: `git status --porcelain` shows nothing
- [ ] CUTLASS file exists: `ls third_party/cutlass/examples/python/CuTeDSL/blackwell/grouped_gemm.py`
- [ ] Backup created: `git branch backup/pre-cutedsl-cherrypick-$(date +%Y%m%d)`

## Cherry-Pick
- [ ] Execute: `git cherry-pick 07fbf93fe555fe105a3a1957263ca8b43b4a14d7`
- [ ] If conflicts: resolve and `git cherry-pick --continue`
- [ ] Verify: `git show HEAD --stat` shows 10 files changed

## Build
- [ ] Clean: `python setup.py clean && rm -rf build/`
- [ ] Build: `USE_CUDA=1 MAX_JOBS=16 python setup.py develop`
- [ ] Verify vendored file: `ls torch/_inductor/kernel/vendored_templates/cutedsl_grouped_gemm.py`
- [ ] Test imports: `python -c "from torch._inductor.template_heuristics.cutedsl import get_groupgemm_configs"`

## Test - Core Functionality
- [ ] Run tests: `python test/inductor/test_cutedsl_grouped_mm.py -v`
  - Expected: PASS on Blackwell GPU OR SKIP on non-Blackwell
- [ ] Basic import: `python -c "import torch; print(torch.__version__)"`
- [ ] CUDA check: `python -c "import torch; print(torch.cuda.is_available())"`

## Test - Integration (if Blackwell GPU available)
- [ ] Create and run integration test (see plan Section 4.2)
- [ ] Run performance benchmark (see plan Section 4.3)
- [ ] Verify no regression: `python test/inductor/test_max_autotune.py -k grouped -v`

## Validate
- [ ] All new files created:
  - `test/inductor/test_cutedsl_grouped_mm.py`
  - `torch/_inductor/kernel/templates/cutedsl_mm_grouped.py.jinja`
  - `torch/_inductor/template_heuristics/cutedsl.py`
- [ ] .gitignore updated with vendored templates entry
- [ ] CI test.sh includes cutedsl test

## Sign-Off
- [ ] Commit message preserved
- [ ] All tests passing (or gracefully skipping)
- [ ] No build warnings related to CuTeDSL
- [ ] Ready for PR/push

---

## Quick Commands

```bash
# 1. Cherry-pick
git cherry-pick 07fbf93fe555fe105a3a1957263ca8b43b4a14d7

# 2. Build
python setup.py clean && USE_CUDA=1 MAX_JOBS=16 python setup.py develop

# 3. Test
python test/inductor/test_cutedsl_grouped_mm.py -v

# 4. Verify
ls torch/_inductor/kernel/vendored_templates/cutedsl_grouped_gemm.py
python -c "from torch._inductor.template_heuristics.cutedsl import get_groupgemm_configs; print('✅ OK')"
```

## Rollback Command
```bash
git revert HEAD  # If already committed
# OR
git reset --hard backup/pre-cutedsl-cherrypick-YYYYMMDD  # Before push
```

---

**Estimated Time:** 2-3 hours (with GPU) | 1 hour (without GPU, verify fallback only)
**See full plan:** `blackwell_cutedsl_cherrypick_plan.md`

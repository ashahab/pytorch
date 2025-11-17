# CuteDSL Import Reference

This document provides the correct import patterns for all CuteDSL-related code in PyTorch v2.8.0 branch.

## ✅ Valid Import Patterns

### From Application Code (mm_scaled_grouped.py)

```python
# Template class
from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate

# Configuration function
from torch._inductor.template_heuristics.cutedsl import get_groupgemm_configs
```

### From CuteDSL Internal Code

```python
# From cutedsl_template.py
from ...autotune_process import CuteDSLBenchmarkRequest, TensorMeta
from ...ir import Buffer, ChoiceCaller, CuteDSLTemplateBuffer, IRNode, Layout, TensorBox
from .cutedsl_kernel import CuteDSLTemplateKernel

# From cutedsl_kernel.py
from .cutedsl_op_overrides import CuteDSLOpOverrides
from ...utils import sympy_index_symbol

# From autotune_process.py
from .codegen.cutedsl.cutedsl_kernel import MAIN_SUFFIX
```

### Package-Level Imports

```python
# Import the module
import torch._inductor.template_heuristics.cutedsl

# Import from package
from torch._inductor.codegen.cutedsl import CuteDSLTemplate
from torch._inductor.codegen.cutedsl import CuteDSLTemplateCaller
```

### Utility Function Imports

```python
from torch._inductor.utils import load_template
from torch._inductor.utils import use_blackwell_cutedsl_grouped_mm
```

## ❌ Invalid Import Patterns

```python
# WRONG - trying to import function as module
import torch._inductor.template_heuristics.cutedsl.get_groupgemm_configs
# ERROR: 'cutedsl' is not a package, get_groupgemm_configs is a function

# CORRECT VERSION:
from torch._inductor.template_heuristics.cutedsl import get_groupgemm_configs
```

## 📁 Package Structure

```
torch/_inductor/
├── codegen/
│   └── cutedsl/              # Package directory
│       ├── __init__.py       # Exports CuteDSLTemplate, CuteDSLTemplateCaller
│       ├── cutedsl_template.py
│       ├── cutedsl_kernel.py
│       ├── cutedsl_op_overrides.py
│       ├── cutedsl_scheduling.py
│       ├── _cutedsl_utils.py
│       └── README.md
├── template_heuristics/
│   ├── __init__.py           # Imports cutedsl module
│   └── cutedsl.py            # Defines get_groupgemm_configs function
├── kernel/
│   ├── mm_scaled_grouped.py  # Uses CuteDSL imports
│   └── templates/
│       └── cutedsl_mm_grouped.py.jinja
├── ir.py                     # Defines CuteDSLTemplateBuffer
├── autotune_process.py       # Defines CuteDSLBenchmarkRequest
└── utils.py                  # Defines load_template, use_blackwell_cutedsl_grouped_mm
```

## 🔍 How to Verify Imports

### Check if a module can be imported:
```bash
python -c "import torch._inductor.template_heuristics.cutedsl; print('OK')"
```

### Check if a function exists:
```bash
python -c "from torch._inductor.template_heuristics.cutedsl import get_groupgemm_configs; print('OK')"
```

### Check if a class exists:
```bash
python -c "from torch._inductor.codegen.cutedsl import CuteDSLTemplate; print('OK')"
```

## 📝 Key Points

1. **cutedsl.py is a module, not a package**
   - Use: `from torch._inductor.template_heuristics.cutedsl import function_name`
   - Don't use: `import torch._inductor.template_heuristics.cutedsl.function_name`

2. **cutedsl/ is a package (has __init__.py)**
   - Use: `from torch._inductor.codegen.cutedsl import ClassName`
   - Or: `from torch._inductor.codegen.cutedsl.module_name import ClassName`

3. **Functions vs Modules**
   - Functions must be imported with `from X import function`
   - Modules can be imported with `import X.module` or `from X import module`

4. **Relative vs Absolute Imports**
   - Internal cutedsl code uses relative imports (`.`, `...`)
   - External code should use absolute imports (`torch._inductor...`)

## ✅ All Imports Verified

All import statements in the codebase have been verified to be correct:
- ✓ mm_scaled_grouped.py imports
- ✓ cutedsl_template.py imports
- ✓ cutedsl_kernel.py imports
- ✓ autotune_process.py imports
- ✓ Package __init__.py imports
- ✓ Utility function imports

The cherry-pick is ready for compilation testing.

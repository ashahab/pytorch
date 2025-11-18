# mypy: allow-untyped-defs
import os

# Configure CUTLASS NVCC architecture for Blackwell (SM 9.0+) and future GPUs
# CRITICAL: This MUST be set before any cutlass imports
if "CUTLASS_NVCC_ARCHS" not in os.environ:
    os.environ["CUTLASS_NVCC_ARCHS"] = "90a-real,100a-real"

from .cutedsl_template import CuteDSLTemplate, CuteDSLTemplateCaller


__all__ = [
    "CuteDSLTemplate",
    "CuteDSLTemplateCaller",
]

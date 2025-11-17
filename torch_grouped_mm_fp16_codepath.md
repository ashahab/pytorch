# FP16 Code Path for `torch._grouped_mm` on CUDA

This document traces the complete execution path when using FP16 input with `torch._grouped_mm` on CUDA GPUs.

## Overview

When you call `torch._grouped_mm` with FP16 tensors on CUDA, it does **NOT** use the optimized CUTLASS grouped GEMM kernel. Instead, it falls back to a sequential loop that calls cuBLAS for each group individually.

## Complete Code Path

### 1. Entry Point: Python API

```python
import torch

# User calls
mat_a = torch.randn(256, 512, dtype=torch.float16, device='cuda')
mat_b = torch.randn(8, 1024, 512, dtype=torch.float16, device='cuda')
offs = torch.tensor([64, 128, 192, 256], dtype=torch.int32, device='cuda')

output = torch._grouped_mm(mat_a, mat_b, offs=offs)
```

### 2. Dispatch to CUDA Implementation

**Location**: `aten/src/ATen/native/cuda/Blas.cpp:1785`

```cpp
Tensor _grouped_mm_cuda(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<c10::ScalarType> out_dtype
) {
    // Step 1: Validate inputs (dimensions, dtypes, alignment)
    _grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);

    // Step 2: Check if fast path is available
    bool a_b_and_out_are_bf16 = (
        mat_a.dtype() == at::kBFloat16 &&
        mat_b.dtype() == at::kBFloat16 &&
        out_dtype.value_or(at::kBFloat16) == at::kBFloat16
    );

    // FP16 check: This will be FALSE for FP16 inputs
    bool use_fast_path = _scaled_mm_allowed_device(/*sm90_only*/true, /*sm100_only*/true)
                         && a_b_and_out_are_bf16;

    // Step 3: Create output tensor
    const auto out_dtype_ = _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
    Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);

    // Step 4: Choose execution path
    if (use_fast_path) {
        // ❌ NOT taken for FP16
        at::cuda::detail::bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
    } else {
        // ✅ FP16 takes this path - FALLBACK
        _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
    }

    return out;
}
```

**Key Decision Point**: Line 1796 - `use_fast_path` is FALSE for FP16 because `a_b_and_out_are_bf16` is FALSE.

### 3. Fallback Implementation

**Location**: `aten/src/ATen/native/GroupedMMUtils.h:115`

```cpp
inline void _grouped_mm_fallback(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<c10::ScalarType> out_dtype,
    Tensor out
) {
    // Log warning message
    LOG(INFO) << "fallback path for `torch._grouped_mm`, performance may not be optimal";

    const bool a_is_2d = mat_a.dim() == 2;
    const bool b_is_2d = mat_b.dim() == 2;

    if (a_is_2d && !b_is_2d) {
        // 2D x 3D case (most common for grouped matmul)
        int group_start_idx = 0;

        // ⚠️ DEVICE-TO-HOST SYNC - Performance bottleneck!
        auto offs_cpu = offs.value().cpu();

        // 🔄 SEQUENTIAL LOOP over groups - NOT parallelized!
        for (int group_idx = 0; group_idx < offs_cpu.size(0); group_idx++) {
            int group_end_idx = offs_cpu[group_idx].item<int>();

            // Slice input for this group
            auto mat_a_slice = mat_a.slice(0, group_start_idx, group_end_idx);
            auto out_slice = out.slice(0, group_start_idx, group_end_idx);

            // 🚀 Individual matmul per group using cuBLAS
            at::mm_out(out_slice, mat_a_slice, mat_b[group_idx]);

            group_start_idx = group_end_idx;
        }
    }
    // ... other cases (3d x 2d, 2d x 2d, 3d x 3d)
}
```

**Performance Issues**:
1. **Device-to-Host sync**: `offs.value().cpu()` - Explicit D2H transfer
2. **Sequential loop**: Each group processed one at a time
3. **Multiple kernel launches**: One cuBLAS call per group
4. **No group-level parallelism**: Cannot leverage GPU's parallel capabilities across groups

### 4. Dispatch to Matrix Multiply

**Location**: `at::mm_out` dispatches to `addmm_out_cuda_impl`

**File**: `aten/src/ATen/native/cuda/Blas.cpp:336`

```cpp
Tensor& addmm_out_cuda_impl(
    Tensor& result,
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Activation activation
) {
    // Validate dimensions and dtypes
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");

    // Check for cuBLASLt availability
    bool useLtInterface = false;

    // For FP16, check if cuBLASLt can be used
    if (!disable_addmm_cuda_lt_final) {
        // cuBLASLt path (lines 385-491)
        // More optimized, uses newer cuBLAS API
        useLtInterface = /* various conditions */;
    }

    if (useLtInterface) {
        // 🚀 Path A: cuBLASLt (newer, more optimized)
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            scalar_type,
            "addmm_cuda_lt",
            [&] {
                at::cuda::blas::gemm_and_bias<scalar_t>(
                    args.transa == 't',
                    args.transb == 't',
                    args.m, args.n, args.k,
                    alpha_val,
                    mat1_ptr, args.mat1_ld,
                    mat2_ptr, args.mat2_ld,
                    self.const_data_ptr<scalar_t>(),
                    result_ptr, args.result_ld,
                    activation_epilogue
                );
            }
        );
    } else {
        // 🚀 Path B: cuBLAS (standard API)
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            scalar_type,
            "addmm_cuda",
            [&] {
                at::cuda::blas::gemm<scalar_t>(
                    args.transa,
                    args.transb,
                    args.m, args.n, args.k,
                    alpha_val,
                    mat1_ptr, args.mat1_ld,
                    mat2_ptr, args.mat2_ld,
                    beta_val,
                    result_ptr, args.result_ld
                );
            }
        );
    }

    return result;
}
```

### 5. cuBLAS Execution

**Location**: `aten/src/ATen/cuda/CUDABlas.cpp:1136-1163`

#### Path 5A: cuBLASLt (Preferred for FP16)

```cpp
bool gemm_and_bias<scalar_t>(/* params */) {
    // Uses cublasLtMatmul API
    // More optimized for modern GPUs
    // Better support for fused operations
    // Can automatically select best algorithm

    cublasLtMatmul(
        handle,
        matmul_desc,
        &alpha,
        A, Adesc,
        B, Bdesc,
        &beta,
        C, Cdesc,
        C, Cdesc,
        &algo,
        workspace, workspace_size,
        stream
    );
}
```

#### Path 5B: cuBLAS Standard (Fallback)

```cpp
template <>
void gemm<at::Half>(/* params */) {
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();

    if (prop->major >= 5) {
        // ✅ Modern GPUs (Maxwell and later)
        // Uses Tensor Cores on SM70+ (Volta, Turing, Ampere, Hopper)

        cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
        if (!at::globalContext().allowFP16ReductionCuBLAS()) {
            cublas_flags = static_cast<cublasMath_t>(
                cublas_flags | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
            );
        }

        cublasSetMathMode(handle, cublas_flags);

        cublasGemmEx(
            handle,
            opa, opb,
            m, n, k,
            &halpha,
            a, CUDA_R_16F, lda,      // FP16 input A
            b, CUDA_R_16F, ldb,      // FP16 input B
            &hbeta,
            c, CUDA_R_16F, ldc,      // FP16 output C
            CUDA_R_16F,              // FP16 compute type
            CUBLAS_GEMM_DEFAULT_TENSOR_OP  // ⭐ TENSOR CORE ENABLED
        );

        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    } else {
        // ❌ Old GPUs (pre-Maxwell)
        cublasSgemmEx(
            handle,
            opa, opb,
            m, n, k,
            &falpha,
            a, CUDA_R_16F, lda,
            b, CUDA_R_16F, ldb,
            &fbeta,
            c, CUDA_R_16F, ldc
        );
    }
}
```

**Key Flag**: `CUBLAS_GEMM_DEFAULT_TENSOR_OP` enables Tensor Core acceleration on supported GPUs.

## Hardware Path Summary

### On SM70+ GPUs (Volta, Turing, Ampere, Hopper, Blackwell)

```
torch._grouped_mm (Python)
    ↓
_grouped_mm_cuda (C++)
    ↓
_grouped_mm_fallback (Sequential loop)
    ↓
[Loop: for each group]
    ↓
    at::mm_out (Per-group matmul)
        ↓
    addmm_out_cuda_impl
        ↓
    at::cuda::blas::gemm<Half> or gemm_and_bias<Half>
        ↓
    cublasGemmEx with CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ↓
    ⚡ TENSOR CORES (Hardware acceleration)
        - FP16 x FP16 → FP16
        - Matrix multiplication in hardware
        - 8x8x4 matrix operations (Volta/Turing)
        - 16x8x16 matrix operations (Ampere+)
```

### On SM50-SM60 GPUs (Maxwell, Pascal)

```
torch._grouped_mm (Python)
    ↓
... (same path until cuBLAS) ...
    ↓
cublasGemmEx (without Tensor Core flag)
    ↓
⚙️ CUDA CORES (Standard SIMD)
    - FP16 x FP16 → FP16
    - SIMD-style multiplication
    - No specialized matrix hardware
```

## Performance Characteristics

### FP16 via Fallback Path

| Aspect | Performance |
|--------|-------------|
| **Group Parallelism** | ❌ None (sequential) |
| **Tensor Cores** | ✅ Yes (SM70+) per group |
| **Kernel Launches** | 🔴 N launches (N = number of groups) |
| **Memory Transfers** | 🔴 D2H sync for offsets |
| **Per-Group Speed** | 🟢 Fast (cuBLAS optimized) |
| **Overall Speed** | 🟡 Moderate (bottlenecked by loop) |

### BF16 via Fast Path (SM90/SM100 only)

| Aspect | Performance |
|--------|-------------|
| **Group Parallelism** | ✅ Full (all groups parallel) |
| **Tensor Cores** | ✅ Yes (CUTLASS optimized) |
| **Kernel Launches** | 🟢 1 launch total |
| **Memory Transfers** | 🟢 No D2H sync |
| **Per-Group Speed** | 🟢 Fast (CUTLASS) |
| **Overall Speed** | 🟢 Excellent |

## Code Path Comparison Table

| Stage | FP16 Path | BF16 Path (SM90/SM100) |
|-------|-----------|------------------------|
| **Entry** | `_grouped_mm_cuda` | `_grouped_mm_cuda` |
| **Decision** | `use_fast_path = false` | `use_fast_path = true` |
| **Implementation** | `_grouped_mm_fallback` | `bf16bf16_grouped_mm` |
| **Loop** | Python-visible sequential loop | GPU kernel internal parallel |
| **D2H Sync** | Yes (`offs.cpu()`) | No |
| **Kernel Type** | cuBLAS per group | CUTLASS grouped GEMM |
| **Tensor Cores** | Yes (if SM70+) | Yes (SM90/SM100) |
| **Memory Layout** | Per-group slicing | Pointer array |
| **Launch Overhead** | High (N launches) | Low (1 launch) |

## Example Execution Timeline

### FP16 (Sequential Fallback)

```
Time →

CPU: [D2H sync]--[launch#1]--[wait]--[launch#2]--[wait]--[launch#3]--[wait]--[launch#4]
                     ↓                    ↓                   ↓                   ↓
GPU:            [group1_mm]          [group2_mm]        [group3_mm]        [group4_mm]
                    ⚡TC                  ⚡TC                ⚡TC                ⚡TC

Total time: T_sync + 4 × (T_launch + T_mm + T_wait)
```

### BF16 (Parallel CUTLASS)

```
Time →

CPU: [launch]--[wait]-----------------------------------[done]
         ↓
GPU:   [group1_mm, group2_mm, group3_mm, group4_mm]  ← All parallel
           ⚡TC        ⚡TC        ⚡TC        ⚡TC

Total time: T_launch + T_mm (parallelized across groups)
```

## Performance Example

```python
import torch
import time

# Problem size
T, B, D_IN, D_OUT, E = 4, 512, 512, 1024, 8

# FP16 (uses fallback)
mat_a = torch.randn(T, B, D_IN, dtype=torch.float16, device='cuda')
mat_b = torch.randn(T, E, D_OUT, D_IN, dtype=torch.float16, device='cuda')
group_sizes = torch.full((T, E), B // E, dtype=torch.int32, device='cuda')

# Benchmark
torch.cuda.synchronize()
start = time.time()

for track_idx in range(T):
    a = mat_a[track_idx].contiguous()
    b = mat_b[track_idx].contiguous()
    offs = group_sizes[track_idx].cumsum(0).to(torch.int32)
    output = torch._grouped_mm(a, b, offs=offs)

torch.cuda.synchronize()
elapsed_fp16 = time.time() - start

print(f"FP16 (fallback): {elapsed_fp16*1000:.2f} ms")
# Expected on A100: ~18-25 ms
# Expected on H100: ~18-25 ms (no fast path for FP16)

# BF16 (uses fast path on H100)
mat_a_bf16 = mat_a.to(torch.bfloat16)
mat_b_bf16 = mat_b.to(torch.bfloat16)

torch.cuda.synchronize()
start = time.time()

for track_idx in range(T):
    a = mat_a_bf16[track_idx].contiguous()
    b = mat_b_bf16[track_idx].contiguous()
    offs = group_sizes[track_idx].cumsum(0).to(torch.int32)
    output = torch._grouped_mm(a, b, offs=offs)

torch.cuda.synchronize()
elapsed_bf16 = time.time() - start

print(f"BF16 (optimized): {elapsed_bf16*1000:.2f} ms")
# Expected on A100: ~18-20 ms (fallback, similar to FP16)
# Expected on H100: ~2-3 ms (fast CUTLASS path!)

print(f"Speedup: {elapsed_fp16/elapsed_bf16:.2f}x")
```

## Recommendations

### For Best Performance

1. **Use BF16 on SM90/SM100 GPUs** (H100, Blackwell)
   - Gets optimized CUTLASS grouped GEMM
   - 5-10x faster than FP16 fallback

2. **Convert FP16 → BF16** if targeting modern GPUs
   ```python
   mat_a_bf16 = mat_a.to(torch.bfloat16)
   output_bf16 = torch._grouped_mm(mat_a_bf16, mat_b_bf16, offs=offs)
   output_fp16 = output_bf16.to(torch.float16)  # Convert back if needed
   ```

3. **On older GPUs (SM80)**: FP16 and BF16 have similar performance
   - Both use fallback path
   - Slight edge to FP16 in some cases

### Understanding the Logs

When you see this message:
```
fallback path for `torch._grouped_mm`, performance may not be optimal
```

It means you're using the sequential loop path described in this document.

## Summary

**FP16 Code Path on CUDA**:
1. ✅ **Functionally works** on all CUDA GPUs
2. ✅ **Uses Tensor Cores** (SM70+) for each group
3. ❌ **Sequential processing** of groups
4. ❌ **Multiple kernel launches** (one per group)
5. ❌ **No CUTLASS optimization**
6. ⚠️ **~2-10x slower** than BF16 on H100

**Bottom Line**: FP16 uses cuBLAS with Tensor Cores but processes groups sequentially. BF16 on H100 uses optimized CUTLASS with parallel group processing.

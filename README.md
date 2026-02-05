# mHC_Offload_CPU_Benchmark

This repository is dedicated to benchmarking the performance of offloading mHC (Manifold-Constrained Hyper-Connections) kernel operations to the CPU. The primary goal is to evaluate the feasibility and latency of moving specific mHC operations (Linear projections, Sinkhorn-Knopp normalization, and Matrix Multiplications) from GPU to CPU, including the overhead of data transfer (host-to-device and device-to-host).

## Overview

The `mHC_benchmark.py` script measures the execution time of the mHC pipeline. When running in "offload simulation" mode (requesting `cpu` device while `cuda` is available), it explicitly benchmarks:

1.  **D2H Copy**: Transferring input tensors from GPU (Device) to CPU (Host).
2.  **Compute**:
    *   **H_res Gen (Linear)**: Linear projections for hyper-connections.
    *   **H_res Gen (Sinkhorn)**: Sinkhorn-Knopp algorithm for normalization.
    *   **h_res App (BMM)**: Application of the hyper-connection matrix (Batch Matrix Multiplication).
3.  **H2D Copy**: Transferring results back from CPU (Host) to GPU (Device).

It also supports comparing the standard PyTorch implementation against optimized kernels provided by `sgl_kernel`.

## Requirements

*   Python 3.x
*   PyTorch
*   (Optional) `sgl_kernel`: For optimized CPU custom ops (Linear, RMSNorm, BMM).

```bash
pip install torch
# Install sgl_kernel if available/needed
```

## Usage

Run the benchmark script `mHC_benchmark.py` with various arguments to configure the model size and execution parameters.

### Arguments

*   `--bs`: Batch size (default: 16).
*   `--sweep_bs`: Sweep batch sizes [1, 2, 4, 8, 16, 32]. Overrides `--bs`.
*   `--seq`: Sequence length (default: 4096).
*   `--dim`: Model dimension (default: 7168).
*   `--n_streams`: Number of streams (default: 4).
*   `--dtype`: Data type. Choices: `float32`, `bfloat16`, `float16`. (default: `bfloat16`).
*   `--profiler`: Enable PyTorch Profiler for detailed traces.

### Examples

**1. Basic Run:**
Benchmark a specific configuration.
```bash
python mHC_benchmark.py --bs 16 --seq 4096 --dim 7168 --dtype bfloat16
```

**2. Sweep Batch Sizes:**
Run benchmarks for batch sizes 1, 2, 4, 8, 16, 32.
```bash
python mHC_benchmark.py --sweep_bs --seq 4096 --dim 7168
```

**3. Enable Profiling:**
Generate PyTorch Profiler traces (saved to `./log/mhc_benchmark`).
```bash
python mHC_benchmark.py --bs 16 --profiler
```

## Output

The script prints the execution time for each step to the console and appends the results to `benchmark_results.csv`.

**Console Output Example:**
```text
Benchmarking on cpu...
...
Offload simulation enabled: Preparing D2H/H2D benchmarks (using Pinned Memory).
  D2H Copy Info: Shape=(16, 4096, 4, 7168), Size=xxx MB
...
------------------ Benchmark Results ----------------
0. D2H Copy (GPU->CPU)             : 12.34 ms | BW: xx GB/s
1. H_res Gen (Linear)              : 5.67 ms
2. H_res Gen (Sinkhorn-Knopp)      : 2.34 ms
3. h_res App (BMM)                 : 8.90 ms
4. H2D Copy (CPU->GPU)             : 10.11 ms | BW: xx GB/s
5. Total                           : 40.45 ms
```

**CSV Columns:**
`Device`, `BatchSize`, `SeqLen`, `Dim`, `NumStreams`, `Dtype`, `SGL_Available`, `Case`, `Time(ms)`

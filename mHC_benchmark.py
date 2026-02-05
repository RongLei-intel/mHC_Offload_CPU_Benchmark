import time
import torch
import sys
import os
import argparse
import csv

sys.path.append(os.path.join(os.getcwd(), 'src'))

from mHC import MHC
from mHC import sinkhorn_knopp

try:
    import sgl_kernel
    from mHC import MHC_Sglang
    SGL_AVAILABLE = True
except ImportError:
    SGL_AVAILABLE = False

def benchmark_op(func, name, n_bytes=None, n_iters=5, warmup=1, use_cuda_events=False):
    for _ in range(warmup): _ = func()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if use_cuda_events and torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(n_iters): _ = func()
        end_event.record()
        
        torch.cuda.synchronize()
        avg_time = start_event.elapsed_time(end_event) / n_iters
    else:
        start = time.perf_counter()
        for _ in range(n_iters): _ = func()
        
        avg_time = (time.perf_counter() - start) / n_iters * 1000

    bw_info = ""
    bw_gbps = None
    if n_bytes is not None:
        # BW in GB/s = (Bytes / 1024^3) / (ms / 1000)
        bw_gbps = (n_bytes / (1024**3)) / (avg_time / 1000)
        bw_info = f" | BW: {bw_gbps:.2f} GB/s"

    print(f"{name:<35}: {avg_time:.4f} ms{bw_info}")
    return avg_time, bw_gbps

def save_results_to_csv(filename, config, results):
    file_exists = os.path.isfile(filename)
    
    # Define fixed fieldnames to ensure alignment across different runs (e.g., CPU vs GPU)
    # This prevents misalignment when some keys (like data transfer) are missing in one run.
    fieldnames = [
        "Device", "BatchSize", "SeqLen", "Dim", "NumStreams", "Dtype", "SGL_Available",
        "Case", "Time(ms)", "Size(MB)", "BW(GB/s)"
    ]
    
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for case_name, metrics in results.items():
            row_data = {**config, "Case": case_name, **metrics}
            writer.writerow(row_data)
    print(f"Results saved to {filename}")

def run_benchmark(device_name, bsz, seq_len, dim, n_streams, data_type, enable_profiler=False):
    print(f"\nBenchmarking on {device_name}...")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available.")
        return
    device = torch.device(device_name)

    # Check if we should simulate offloading (CPU run with GPU available)
    simulate_offload = (device_name == "cpu" and torch.cuda.is_available())
    
    print(f"Config: B={bsz}, T={seq_len}, C={dim}, N={n_streams}, Dtype={data_type}")
    
    # Store configuration
    config_info = {
        "Device": device_name,
        "BatchSize": bsz,
        "SeqLen": seq_len,
        "Dim": dim,
        "NumStreams": n_streams,
        "Dtype": str(data_type),
        "SGL_Available": SGL_AVAILABLE
    }
    results_info = {}

    if SGL_AVAILABLE:
        mhc_layer = MHC_Sglang(dim=dim, n_streams=n_streams, data_type=data_type).to(device).eval()
        print("Using SGL kernel for mHC.")
    else:
        mhc_layer = MHC(dim=dim, n_streams=n_streams, data_type=data_type).to(device).eval()
        print("Using standard mHC implementation.")
    
    # Initialize CPU tensors
    x_streams = mhc_layer.init_streams(torch.randn(bsz, seq_len, dim, device=device, dtype=data_type))
    h_res_output = torch.empty((bsz * seq_len, n_streams, dim), device=device, dtype=data_type)

    x_streams_gpu = None
    d2h_bytes = 0
    h2d_bytes = 0

    if simulate_offload:
        # Move source to GPU to simulate starting point
        x_streams_gpu = x_streams.to("cuda")
        
        # Optimize CPU buffers: Use pinned memory for faster D2H/H2D
        x_streams = x_streams.pin_memory()
        h_res_output = h_res_output.pin_memory()
        
        d2h_bytes = x_streams.numel() * x_streams.element_size()
        h2d_bytes = h_res_output.numel() * h_res_output.element_size()

        print("\nOffload simulation enabled: Preparing D2H/H2D benchmarks (using Pinned Memory).")
        print(f"  D2H Copy Info: Shape={tuple(x_streams.shape)}, Size={d2h_bytes/1024/1024:.2f} MB")
        print(f"  H2D Copy Info: Shape={tuple(h_res_output.shape)}, Size={h2d_bytes/1024/1024:.2f} MB")

    use_cuda_events = (device_name == "cuda")

    def print_io_info(case_name, inputs, output):
        print(f"\n[{case_name}] Info:")
        for i, inp in enumerate(inputs):
            print(f"  Input {i}: Shape={tuple(inp.shape)}, Device={inp.device}, Dtype={inp.dtype}")
        print(f"  Output : Shape={tuple(output.shape)}, Device={output.device}, Dtype={output.dtype}")

    # 1. H_res Generation (RMSNorm + Linear)
    calc_linear = lambda: mhc_layer.cal_H_res(x_streams, SGL_AVAILABLE)
    with torch.no_grad(): h_res_logits = calc_linear()
    print_io_info("1. H_res Gen (RMSNorm + Linear)", [x_streams], h_res_logits)
    
    # 2. Sinkhorn
    calc_sinkhorn = lambda: mhc_layer.cal_sinkhorn(h_res_logits)
    with torch.no_grad(): h_res_matrix = calc_sinkhorn()
    print_io_info("2. H_res Gen (Sinkhorn-Knopp)", [h_res_logits], h_res_matrix)
    
    # 3. bmm
    if SGL_AVAILABLE:
        app_x_streams = x_streams.clone()
        app_x_streams = app_x_streams.reshape(bsz * seq_len, n_streams, dim).permute(0, 2, 1).contiguous()
    else:
        app_x_streams = x_streams
    calc_app = lambda: mhc_layer.cal_app(h_res_matrix, app_x_streams, SGL_AVAILABLE, out=h_res_output)
    with torch.no_grad(): app_out = calc_app()
    print_io_info("3. h_res App (BMM)", [h_res_matrix, app_x_streams], h_res_output if SGL_AVAILABLE else app_out)
    
    # 4. Total
    if simulate_offload:
        def calc_total():
            # D2H: GPU -> CPU (Pinned)
            # copy_(..., non_blocking=True) is essentially asynchronous 
            x_streams.copy_(x_streams_gpu, non_blocking=True)
            
            # Compute on CPU
            mhc_layer.cal_total(x_streams, app_x_streams, SGL_AVAILABLE, out=h_res_output)
            
            # H2D: CPU (Pinned) -> GPU
            return h_res_output.to("cuda", non_blocking=False)
        
        with torch.no_grad(): total_out = calc_total()
        # Synchronize for correct timing if non_blocking=True was used, 
        # though benchmark_op handles timing, explicit sync helps logical correctness verification
        torch.cuda.synchronize() 
        print_io_info("4. Total (D2H + CPU Compute + H2D)", [x_streams_gpu], total_out)
    else:
        calc_total = lambda: mhc_layer.cal_total(x_streams, app_x_streams, SGL_AVAILABLE, out=h_res_output)
        
        with torch.no_grad(): total_out = calc_total()
        print_io_info("4. Total", [x_streams], h_res_output if SGL_AVAILABLE else total_out)
    
    # 0 & 5. D2H and H2D Transfer Lambdas
    calc_d2h = None
    calc_h2d = None
    if simulate_offload:
        # Optimized transfers
        calc_d2h = lambda: x_streams.copy_(x_streams_gpu, non_blocking=False)
        calc_h2d = lambda: h_res_output.to("cuda", non_blocking=False)

    profiler = None
    if enable_profiler:
        print("Profiling enabled...")
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=0),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/mhc_benchmark'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()
    
    print("------------------ Benchmark Results ----------------")
    with torch.inference_mode():
        if simulate_offload:
            t, bw = benchmark_op(calc_d2h, "0. D2H Copy (GPU->CPU)", n_bytes=d2h_bytes, use_cuda_events=True)
            results_info["0. D2H Copy"] = {"Time(ms)": t, "Size(MB)": d2h_bytes / (1024 * 1024), "BW(GB/s)": bw}
            if profiler: profiler.step()
            time.sleep(1)

        t, _ = benchmark_op(calc_linear, "1. H_res Gen (Linear)", use_cuda_events=use_cuda_events)
        results_info["1. H_res Gen (Linear)"] = {"Time(ms)": t}
        if profiler: profiler.step()
        time.sleep(1)

        t, _ = benchmark_op(calc_sinkhorn, "2. H_res Gen (Sinkhorn-Knopp)", use_cuda_events=use_cuda_events)
        results_info["2. H_res Gen (Sinkhorn-Knopp)"] = {"Time(ms)": t}
        if profiler: profiler.step()
        time.sleep(1)

        t, _ = benchmark_op(calc_app, "3. h_res App (BMM)", use_cuda_events=use_cuda_events)
        results_info["3. h_res App (BMM)"] = {"Time(ms)": t}
        if profiler: profiler.step()
        time.sleep(1)

        if simulate_offload:
            t, bw = benchmark_op(calc_h2d, "4. H2D Copy (CPU->GPU)", n_bytes=h2d_bytes, use_cuda_events=True)
            results_info["4. H2D Copy"] = {"Time(ms)": t, "Size(MB)": h2d_bytes / (1024 * 1024), "BW(GB/s)": bw}
            if profiler: profiler.step()
            time.sleep(1)

        t, _ = benchmark_op(calc_total, "5. Total", use_cuda_events=use_cuda_events)
        results_info["5. Total"] = {"Time(ms)": t}
        if profiler: profiler.step()
        time.sleep(1)

    if profiler:
        profiler.stop()
        print("Profiling done. Results saved to ./log/mhc_benchmark")

    save_results_to_csv("benchmark_results.csv", config_info, results_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=16, help='Batch size (ignored if --sweep_bs is used)')
    parser.add_argument('--sweep_bs', action='store_true', help='Sweep batch sizes from 1 to 16 (powers of 2)')
    parser.add_argument('--seq', type=int, default=4096, help='Sequence length')
    parser.add_argument('--dim', type=int, default=7168, help='Model dimension')
    parser.add_argument('--n_streams', type=int, default=4, help='Number of streams')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'], help='Data type')
    parser.add_argument('--profiler', action='store_true', help='Enable PyTorch profiler')
    args = parser.parse_args()

    dtype_map = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }
    data_type = dtype_map[args.dtype]

    if args.sweep_bs:
        batch_sizes = [1, 2, 4, 8, 16]
    else:
        batch_sizes = [args.bs]

    for bs in batch_sizes:
        run_benchmark("cpu", bs, args.seq, args.dim, args.n_streams, data_type, args.profiler)
        if torch.cuda.is_available(): 
            run_benchmark("cuda", bs, args.seq, args.dim, args.n_streams, data_type, args.profiler)
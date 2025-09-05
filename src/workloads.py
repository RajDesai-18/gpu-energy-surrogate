"""
MatMul sweep runner (GPU) for dataset bootstrap.

- Runs square GEMM (N x N) on CUDA via PyTorch.
- For each N: 3 warmups + 3 timed iterations using CUDA events.
- Prints average runtime_ms and achieved GFLOP/s.
- Writes ONE CSV per size under data/raw/ with schema fields we’ll extend later.

Columns (initial):
  workload, N, dtype, repeats, runtime_ms, flops_G, gflops, times_ms_json,
  timestamp, avg_power_W, energy_J, power_series_path, counters_json,
  device_name, torch_version, cuda_version, os
"""

import argparse
import json
import os
import platform
from datetime import datetime, timezone

import torch
import pandas as pd

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def dtype_from_str(s: str):
    s = s.lower()
    if s in ("fp32", "float32", "f32"):
        return torch.float32, "float32"
    if s in ("fp16", "float16", "f16", "half"):
        return torch.float16, "float16"
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16, "bfloat16"
    raise ValueError(f"Unsupported dtype: {s}")

def run_matmul(N: int, dtype: torch.dtype, repeats: int = 3, warmups: int =3, device: str = "cuda"):
    assert torch.cuda.is_available(), "CUDA not available. Check your install."
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    try: 
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    A = torch.randn((N,N), device=device, dtype=dtype)
    B = torch.randn((N,N), device=device, dtype=dtype)

    for _ in range(warmups):
        C = A @ B
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        C = A @ B
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    avg_ms = sum(times_ms) / len(times_ms)

    flops = 2.0 * (N ** 3)
    flops_G = flops / 1e9
    gflops_achieved = flops_G / (avg_ms / 1000.0)

    return avg_ms, flops_G, gflops_achieved, times_ms

def wrtie_csv_row(outdir, rowdict):
    ensure_dir(outdir)
    now_utc = datetime.now(timezone.utc)
    ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
    fname = f"matmul_N{rowdict['N']}_{rowdict['dtype']}_{ts}.csv"
    fpath = os.path.join(outdir, fname)
    pd.DataFrame([rowdict]).to_csv(fpath, index=False)
    return fpath

def main():
    parser = argparse.ArgumentParser(description="GPU MatMul sweep (PyTorch)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048], help="List of sqaure size N to run (e.g., --sizes 512 1024 2048)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "fp32", "float16", "fp16", "half", "bf16", "bfloat16"])
    parser.add_argument("--repeats", type=int, default=3, help="Timed iterations")
    parser.add_argument("--warmups", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--outdir", type=str, default="data/raw", help="Output CSV directory")
    args = parser.parse_args()
    now_utc = datetime.now(timezone.utc)

    torch_dtype, dtype_str = dtype_from_str(args.dtype)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    os_name = f"{platform.system()} {platform.release()}"

    for N in args.sizes:
        avg_ms, flops_G, gflops, times_ms = run_matmul(
            N=N, dtype=torch_dtype, repeats=args.repeats, warmups=args.warmups, device="cuda"
        )

        row = {
            "workload": "matmul",
            "N": N,
            "dtype": dtype_str,
            "repeats": args.repeats,
            "runtime_ms": round(avg_ms, 6),
            "flops_G": round(flops_G, 6),
            "gflops": round(gflops, 6),
            "times_ms_json": json.dumps([round(t, 6) for t in times_ms]),
            "timestamp": now_utc.isoformat().replace("+00:00", "Z"),

            # Placeholders for future steps
            "avg_power_W": None,
            "energy_J": None,
            "power_series_path": "",
            "counters_json": "{}",  # will fill with Nsight Compute later

            # Environment info
            "device_name": device_name,
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "os": os_name,
        }

        fpath = wrtie_csv_row(args.outdir, row)
        print(
            f"[matmul] N={N} dtype={dtype_str} avg_runtime={row['runtime_ms']} ms "
            f"achieved={row['gflops']} GFLOP/s -> {fpath}"
        )

if __name__ == "__main__":
    main()

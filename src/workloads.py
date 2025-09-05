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
from src.power_log import sample_block


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


def run_matmul(
    N: int, dtype: torch.dtype, repeats: int = 3, warmups: int = 3, device: str = "cuda"
):
    assert torch.cuda.is_available(), "CUDA not available. Check your install."
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    A = torch.randn((N, N), device=device, dtype=dtype)
    B = torch.randn((N, N), device=device, dtype=dtype)

    for _ in range(warmups):
        A @ B
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        A @ B
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    avg_ms = sum(times_ms) / len(times_ms)

    flops = 2.0 * (N**3)
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
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
        help="List of sqaure size N to run (e.g., --sizes 512 1024 2048)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "fp32", "float16", "fp16", "half", "bf16", "bfloat16"],
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timed iterations")
    parser.add_argument("--warmups", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--outdir", type=str, default="data/raw", help="Output CSV directory")
    parser.add_argument(
        "--with-power", action="store_true", help="Sample power during the timed block"
    )
    parser.add_argument(
        "--power-interval-ms",
        type=float,
        default=20.0,
        help="Power sampling interval in ms",
    )
    parser.add_argument(
        "--power-min-runtime-ms",
        type=float,
        default=500.0,
        help="Ensure at least this much compute time is captured when --with-power is used",
    )
    args = parser.parse_args()
    now_utc = datetime.now(timezone.utc)

    torch_dtype, dtype_str = dtype_from_str(args.dtype)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    os_name = f"{platform.system()} {platform.release()}"

    for N in args.sizes:
        label = f"matmul_N{N}_{dtype_str}"
        k_repeats_for_power = 1  # will be updated inside the closure

        def _timed_block_k(N=N, torch_dtype=torch_dtype):
            # First run: get timings we will report
            avg_ms_1, flops_G_1, gflops_1, times_ms_1 = run_matmul(
                N=N,
                dtype=torch_dtype,
                repeats=args.repeats,
                warmups=args.warmups,
                device="cuda",
            )
            nonlocal k_repeats_for_power
            # Compute how many runs we need to reach the target power window
            target_ms = max(1.0, float(args.power_min_runtime_ms))
            k_repeats_for_power = max(1, int((target_ms + avg_ms_1 - 1e-9) // max(1e-6, avg_ms_1)))
            # Do the remaining runs (no warmups) to extend the compute window
            for _ in range(k_repeats_for_power - 1):
                run_matmul(
                    N=N,
                    dtype=torch_dtype,
                    repeats=args.repeats,
                    warmups=0,
                    device="cuda",
                )
            return avg_ms_1, flops_G_1, gflops_1, times_ms_1

        if args.with_power:
            (
                (avg_ms, flops_G, gflops, times_ms),
                avg_power_W_total,
                energy_J_total,
                power_path,
            ) = sample_block(
                label=label,
                fn=_timed_block_k,
                outdir="data/raw/power",
                device_index=0,
                interval_s=args.power_interval_ms / 1000.0,
            )
            # Convert total energy to per-run energy for the CSV (so rows remain comparable)
            energy_J = energy_J_total / float(k_repeats_for_power)
            avg_power_W = avg_power_W_total  # average over the whole sampled window
        else:
            avg_ms, flops_G, gflops, times_ms = run_matmul(
                N=N,
                dtype=torch_dtype,
                repeats=args.repeats,
                warmups=args.warmups,
                device="cuda",
            )
            avg_power_W, energy_J, power_path = None, None, ""

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
            "avg_power_W": avg_power_W,
            "energy_J": energy_J,
            "power_series_path": power_path,
            "counters_json": "{}",  # will fill with Nsight Compute later
            # Environment info
            "device_name": device_name,
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "os": os_name,
        }

        fpath = wrtie_csv_row(args.outdir, row)
        msg = (
            f"[matmul] N={N} dtype={dtype_str} avg_runtime={row['runtime_ms']} ms "
            f"achieved={row['gflops']} GFLOP/s"
        )
        if args.with_power and avg_power_W is not None:
            msg += f" | avg_power={avg_power_W} W energy_per_run={energy_J} J"
        msg += f" -> {fpath}"
        print(msg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Power sampler with dual backend:
- NVML (pynvml) if available
- Fallback to `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms <ms> -i 0`

Writes CSV: time_s,power_W and exposes avg_power_W, energy_J, power_series_path.
"""

import os
import csv
import time
import threading
import subprocess
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone


# ----- Optional: help Windows find nvml.dll if present -----
def _win_hint_nvml():
    if os.name != "nt":
        return
    try:
        import ctypes
        from pathlib import Path as _P

        windir = os.environ.get("WINDIR", r"C:\Windows")
        sys32 = _P(windir) / "System32"
        dll = sys32 / "nvml.dll"
        if dll.exists():
            try:
                os.add_dll_directory(str(sys32))
            except Exception:
                os.environ["PATH"] = str(sys32) + os.pathsep + os.environ.get("PATH", "")
            try:
                ctypes.CDLL(str(dll))
            except Exception:
                pass
            return
        ds = sys32 / "DriverStore" / "FileRepository"
        if ds.is_dir():
            for p in ds.rglob("nvml.dll"):
                d = str(p.parent)
                try:
                    os.add_dll_directory(d)
                except Exception:
                    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                try:
                    ctypes.CDLL(str(p))
                except Exception:
                    pass
                return
    except Exception:
        pass


_win_hint_nvml()

# ----- Try NVML import (might fail; we’ll fall back) -----
_nvml_ok = False
try:
    import pynvml as nvml

    try:
        nvml.nvmlInit()
        nvml.nvmlShutdown()
        _nvml_ok = True
    except Exception:
        _nvml_ok = False
except Exception:
    _nvml_ok = False


@dataclass
class Sample:
    t: float  # seconds since start
    p: float  # watts


class PowerSampler:
    def __init__(
        self,
        label: str,
        outdir: str = "data/raw/power",
        device_index: int = 0,
        interval_s: float = 0.02,
    ):
        self.label = label
        self.outdir = Path(outdir)
        self.device_index = device_index
        self.interval_s = interval_s
        self._running = False
        self._thread = None
        self._t0 = None
        self._samples = []
        self.avg_power_W = None
        self.energy_J = None
        self.power_series_path = ""
        # pick backend
        self.backend = "nvml" if _nvml_ok else "nvidia-smi"
        self._proc = None  # for nvidia-smi

    # ---------- NVML backend loop ----------
    def _loop_nvml(self, handle):
        next_t = self.interval_s
        while self._running:
            try:
                p_w = float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
            except Exception:
                p_w = 0.0
            t = time.perf_counter() - self._t0
            self._samples.append(Sample(t=t, p=p_w))
            time.sleep(max(0.0, next_t - t))
            next_t += self.interval_s

    # ---------- nvidia-smi backend loop ----------
    def _loop_smi(self, exe):
        # stream numbers (W) line-by-line
        # example cmd: nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 20 -i 0
        args = [
            exe,
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
            "-lms",
            str(int(max(1, round(self.interval_s * 1000)))),
            "-i",
            str(self.device_index),
        ]
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            bufsize=1,
        )
        while self._running and self._proc and self._proc.stdout:
            line = self._proc.stdout.readline()
            if not line:
                break
            try:
                p_w = float(line.strip())
            except Exception:
                continue
            t = time.perf_counter() - self._t0
            self._samples.append(Sample(t=t, p=p_w))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def start(self):
        self._samples.clear()
        self._t0 = time.perf_counter()
        self._running = True

        if self.backend == "nvml":
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._thread = threading.Thread(target=self._loop_nvml, args=(handle,), daemon=True)
        else:
            # find nvidia-smi
            exe = "nvidia-smi"
            if os.name == "nt":
                exe = os.path.join(
                    os.environ.get("WINDIR", r"C:\Windows"),
                    "System32",
                    "nvidia-smi.exe",
                )
            self._thread = threading.Thread(target=self._loop_smi, args=(exe,), daemon=True)

        self._thread.start()

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        # Stop/cleanup backend
        if self.backend == "nvml":
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
        else:
            if self._proc:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                try:
                    self._proc.wait(timeout=1.0)
                except Exception:
                    pass

        # Final sample if needed
        if len(self._samples) < 2:
            # Try one last read
            if self.backend == "nvml":
                try:
                    h = nvml.nvmlDeviceGetHandleByIndex(self.device_index)
                    p_w = float(nvml.nvmlDeviceGetPowerUsage(h)) / 1000.0
                except Exception:
                    p_w = 0.0
            else:
                p_w = 0.0
            t = time.perf_counter() - self._t0
            self._samples.append(Sample(t=t, p=p_w))

        # Integrate energy (trapezoid)
        energy = 0.0
        for i in range(1, len(self._samples)):
            p0, t0 = self._samples[i - 1].p, self._samples[i - 1].t
            p1, t1 = self._samples[i].p, self._samples[i].t
            energy += 0.5 * (p0 + p1) * (t1 - t0)
        duration = max(1e-9, self._samples[-1].t - self._samples[0].t)
        avg_power = energy / duration

        # Write CSV
        self.outdir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fname = f"{self.label}_{self.backend}_{ts}.csv"
        fpath = self.outdir / fname
        with fpath.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "power_W"])
            for s in self._samples:
                w.writerow([f"{s.t:.6f}", f"{s.p:.6f}"])

        self.power_series_path = str(fpath).replace("\\", "/")
        self.avg_power_W = float(f"{avg_power:.6f}")
        self.energy_J = float(f"{energy:.6f}")


def sample_block(
    label: str,
    fn,
    outdir: str = "data/raw/power",
    device_index: int = 0,
    interval_s: float = 0.02,
):
    """
    Run `fn()` while sampling power. Returns (result, avg_power_W, energy_J, power_series_path).
    """
    with PowerSampler(
        label=label, outdir=outdir, device_index=device_index, interval_s=interval_s
    ) as ps:
        result = fn()
    return result, ps.avg_power_W, ps.energy_J, ps.power_series_path

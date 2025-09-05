# GPU Kernel Energy Surrogate

> Predict kernel **power (W)** and **energy (J)** from inputs + counters. Compare **Early** (params-only) vs **Counter** (params + hardware counters) models. Demonstrate a perf/Watt decision.

**Status:** Work in Progress

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/RajDesai-18/gpu-energy-surrogate.git
cd gpu-energy-surrogate
````

### 2. Create & activate environment

Using conda:

```bash
conda create -n energy python=3.10 -y
conda activate energy
```

Or with venv:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install PyTorch separately for your CUDA version:
👉 [PyTorch Install Guide](https://pytorch.org/get-started/locally/)

Example (CUDA 12.1):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4. Run a quick smoke test

```bash
python - << "PY"
print("Repo is set up correctly ✅")
PY
```

---

## Project Layout

```text
gpu-energy-surrogate/
├─ notebooks/
│  ├─ 01_collect.ipynb         # data collection
│  └─ 02_model.ipynb           # modeling (Early vs Counter)
├─ src/
│  ├─ workloads.py             # workloads & runner
│  ├─ power_log.py             # NVML/pyRAPL power sampling
│  ├─ counters.py              # Nsight Compute / perf wrappers
│  └─ features.py              # feature assembly
├─ data/
│  ├─ raw/                     # raw logs (not committed)
│  └─ processed/
│     └─ dataset.csv           # final dataset (committed)
├─ .github/workflows/ci.yml    # lint + smoke test
├─ energy_cli.py               # predict power/energy for one row
├─ report.md                   # 2–3 page summary
├─ requirements.txt            # minimal deps (PyTorch separate)
└─ README.md
```

---

## Workflow

* **Branching model:** feature branches (`feat/*`, `fix/*`, `chore/*`)
* **Commits:** Conventional Commits (`feat(scope): ...`)
* **CI:** GitHub Actions (black + flake8 + smoke) runs on PRs & pushes
* **Main branch:** protected, requires PR + passing CI before merge

---

## Current Features

* ✅ MatMul sweep workload (`src/workloads.py`)
* ✅ CI setup with lint + smoke test
* 🚧 NVML power logger (`src/power_log.py`)
* 🚧 Nsight counters collection
* 🚧 Dataset merge → `data/processed/dataset.csv`
* 🚧 Early vs Counter ML models (notebooks)
* 🚧 CLI for predictions (`energy_cli.py`)
* 🚧 Perf/Watt demo and report

---

## Contributing

This is a solo project, but branches + PRs are used for clean history and CI gating.

Typical flow:

```bash
git checkout -b feat/new-feature
# edit files
git commit -m "feat(scope): description"
git push -u origin feat/new-feature
gh pr create --title "feat(scope): description" --body "details..."
```

---

## Badges

[![CI](https://github.com/RajDesai-18/gpu-energy-surrogate/actions/workflows/ci.yml/badge.svg)](https://github.com/RajDesai-18/gpu-energy-surrogate/actions)

```



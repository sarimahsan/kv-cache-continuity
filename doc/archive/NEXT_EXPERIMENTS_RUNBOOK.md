# Next Experiments Runbook

**Project:** KV-cache continuity and self-speculative decoding  
**Updated:** 2026-10-15  
**Primary model:** `Qwen/Qwen2.5-3B`  
**Execution target:** NVIDIA GPU (Kaggle GPU or local CUDA machine)

## 1. Current status

The core KV-continuity evidence is complete. The remaining scheduled work is the
optional **real-speedup improvement track**:

1. Validate multi-token speculative decoding with `gamma=2`.
2. Train the lightweight tuned-lens readout adapter.
3. Sweep draft depth versus acceptance and measured throughput.
4. Repeat the publication-scale 60-prompt benchmark using the tuned lens.
5. Compare the tuned lens against the existing MLP adapter and update the paper
   only if the comparison is reproducible.

Do not rerun the completed continuous Qwen sweep, LLaMA sweep, or adapter-capacity
ablation unless a result file is found to be corrupted.

## 2. Expected outputs

Use these output locations so results remain separated from source files:

```text
checkpoints/exit_adapters_qwen_tuned_lens.pt
results/canonical_sweeps/qwen2.5_3b/speculative_depth_sweep.csv
results/canonical_sweeps/qwen2.5_3b/speculative_depth_tradeoff.pdf
results/canonical_sweeps/qwen2.5_3b/benchmark_tuned_lens_results.json
results/raw_runs/<run-name>/
```

The `gamma=2` smoke benchmark currently prints its summary to stdout. Save the
terminal output in its raw-run directory.

## 3. Run order

Run the experiments in this order:

| Order | Experiment | Why it comes next | Pass condition |
|---:|---|---|---|
| 0 | Environment and smoke check | Prevent wasting GPU time on a broken setup | Model loads and one short run completes |
| 1 | `gamma=2` smoke benchmark | Checks lossless multi-token verification | No crash/NaN; record exact-match rate |
| 2 | Tuned-lens training | Creates the missing readout checkpoint | Checkpoint exists and training completes |
| 3a | Existing MLP depth sweep | Baseline for a fair depth comparison | CSV and PDF are produced |
| 3b | Tuned-lens depth sweep | Measures whether the new adapter improves the trade-off | Same outputs produced with tuned-lens results |
| 4 | Tuned-lens 60-prompt benchmark | Publication-scale confirmation | Report mean speedup, CI, acceptance, exact-match rate |
| 5 | Analysis and manuscript update | Prevents claims based on one noisy run | Keep only reproducible improvements |

## 4. Local Windows setup

Run these commands from the repository root in PowerShell:

```powershell
Set-Location D:\CUDA-Graph\kv_continuity

New-Item -ItemType Directory -Force -Path `
  results\raw_runs\gamma2_smoke, `
  results\raw_runs\tuned_lens_training, `
  results\canonical_sweeps\qwen2.5_3b | Out-Null

python -m pip install -r requirements.txt
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If the CUDA check prints `False`, stop rather than running the 3B experiments on
CPU.

## 5. Experiment 1: `gamma=2` lossless smoke benchmark

This uses the already validated MLP checkpoint and tests multi-token drafting
before introducing a new adapter.

```powershell
Set-Location D:\CUDA-Graph\kv_continuity

$run = "results\raw_runs\gamma2_smoke\stdout.txt"
python experiments\benchmark_speculative_decoding.py `
  --model-id Qwen/Qwen2.5-3B `
  --adapter-path checkpoints\exit_adapters_qwen_mlp.pt `
  --draft-depth 24 `
  --gamma 2 `
  --max-tokens 40 `
  --dtype float16 `
  --device cuda 2>&1 | Tee-Object -FilePath $run
```

Record from the output:

- Average speedup.
- Average acceptance rate.
- Exact token-match count.
- Any CUDA out-of-memory, NaN, or shape errors.

If this fails, do not proceed to the tuned-lens benchmark. Preserve the complete
log and fix the failure first.

## 6. Experiment 2: train the tuned-lens adapter

```powershell
Set-Location D:\CUDA-Graph\kv_continuity

New-Item -ItemType Directory -Force -Path `
  checkpoints, `
  results\raw_runs\tuned_lens_training | Out-Null

python experiments\train_tuned_lens.py `
  --model-id Qwen/Qwen2.5-3B `
  --target-depths 14,18,22,24,28 `
  --adapter-type lora `
  --rank 64 `
  --steps 400 `
  --batch-size 2 `
  --seq-len 128 `
  --lr 1e-3 `
  --distill-temp 2.0 `
  --output-path checkpoints\exit_adapters_qwen_tuned_lens.pt `
  --device cuda 2>&1 | Tee-Object -FilePath results\raw_runs\tuned_lens_training\stdout.txt
```

Verify the checkpoint before starting the sweep:

```powershell
Test-Path checkpoints\exit_adapters_qwen_tuned_lens.pt
Get-Item checkpoints\exit_adapters_qwen_tuned_lens.pt | Select-Object FullName,Length,LastWriteTime
```

The checkpoint must exist and have a non-zero size. Keep the training log with
the checkpoint.

## 7. Experiment 3a: baseline MLP depth sweep

Run the existing adapter first so the new comparison has a baseline:

```powershell
Set-Location D:\CUDA-Graph\kv_continuity

python experiments\sweep_speculative_depths.py `
  --model-id Qwen/Qwen2.5-3B `
  --adapter-path checkpoints\exit_adapters_qwen_mlp.pt `
  --depths 14,18,22,24,28,30 `
  --gamma 1 `
  --num-prompts 12 `
  --max-tokens 30 `
  --dtype float16 `
  --device cuda `
  --output-dir results\canonical_sweeps\qwen2.5_3b
```

If the script overwrites the same CSV/PDF on each run, copy the baseline files
before running the tuned-lens sweep:

```powershell
Copy-Item results\canonical_sweeps\qwen2.5_3b\speculative_depth_sweep.csv `
  results\canonical_sweeps\qwen2.5_3b\speculative_depth_sweep_mlp.csv
Copy-Item results\canonical_sweeps\qwen2.5_3b\speculative_depth_tradeoff.pdf `
  results\canonical_sweeps\qwen2.5_3b\speculative_depth_tradeoff_mlp.pdf
```

## 8. Experiment 3b: tuned-lens depth sweep

```powershell
Set-Location D:\CUDA-Graph\kv_continuity

python experiments\sweep_speculative_depths.py `
  --model-id Qwen/Qwen2.5-3B `
  --adapter-path checkpoints\exit_adapters_qwen_tuned_lens.pt `
  --depths 14,18,22,24,28,30 `
  --gamma 1 `
  --num-prompts 12 `
  --max-tokens 30 `
  --dtype float16 `
  --device cuda `
  --output-dir results\canonical_sweeps\qwen2.5_3b
```

Save the tuned-lens output under distinct names if necessary:

```powershell
Copy-Item results\canonical_sweeps\qwen2.5_3b\speculative_depth_sweep.csv `
  results\canonical_sweeps\qwen2.5_3b\speculative_depth_sweep_tuned_lens.csv
Copy-Item results\canonical_sweeps\qwen2.5_3b\speculative_depth_tradeoff.pdf `
  results\canonical_sweeps\qwen2.5_3b\speculative_depth_tradeoff_tuned_lens.pdf
```

Compare the same depths between adapters. Do not compare different prompt
counts, token budgets, dtypes, or temperatures.

## 9. Experiment 4: tuned-lens 60-prompt benchmark

```powershell
Set-Location D:\CUDA-Graph\kv_continuity

python experiments\benchmark_large_scale.py `
  --model-id Qwen/Qwen2.5-3B `
  --adapter-path checkpoints\exit_adapters_qwen_tuned_lens.pt `
  --draft-depth 24 `
  --gamma 1 `
  --num-prompts-per-domain 15 `
  --max-tokens 30 `
  --dtype float16 `
  --device cuda `
  --output-json results\canonical_sweeps\qwen2.5_3b\benchmark_tuned_lens_results.json
```

Check the JSON:

```powershell
Get-Content results\canonical_sweeps\qwen2.5_3b\benchmark_tuned_lens_results.json
```

Report at minimum:

- Mean speedup and 95% confidence interval.
- Mean acceptance rate and 95% confidence interval.
- Exact-match count and percentage.
- Peak VRAM.
- Number of prompts and prompts per domain.

For a credible improvement claim, the tuned-lens run must use the same model,
depth, horizon, prompt count, token count, dtype, and device as the existing
baseline JSON.

## 10. Kaggle procedure

### 10.1 Create the notebook

In Kaggle:

1. Create a new **Notebook**.
2. Select **GPU T4 x2** if available; otherwise select a GPU accelerator.
3. Turn **Internet** on so Hugging Face can download `Qwen/Qwen2.5-3B`.
4. Add the GitHub repository as a notebook input, or clone it in the first cell.

### 10.2 Clone and install

Use a Kaggle notebook cell:

```bash
%cd /kaggle/working
!rm -rf kv-cache-continuity
!git clone https://github.com/sarimahsan/kv-cache-continuity.git
%cd /kaggle/working/kv-cache-continuity
!pip install -q -r requirements.txt
!mkdir -p results/raw_runs/gamma2_smoke results/raw_runs/tuned_lens_training results/canonical_sweeps/qwen2.5_3b
!nvidia-smi
```

If the repository is already attached as a Kaggle dataset, replace the clone
step with:

```bash
%cd /kaggle/input/kv-cache-continuity
```

However, write outputs to `/kaggle/working/kv-cache-continuity`, not to the
read-only `/kaggle/input` directory.

### 10.3 Optional Hugging Face authentication

The Qwen model may be downloadable without authentication. If Kaggle reports a
Hugging Face authorization or rate-limit error, create a Kaggle Secret named
`HF_TOKEN`, then run:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Never paste the token into a notebook cell or commit it to the repository.

### 10.4 Run the Kaggle experiments

Run each command in a separate notebook cell. The `tee` commands preserve logs.

#### Smoke benchmark

```bash
%cd /kaggle/working/kv-cache-continuity
!python experiments/benchmark_speculative_decoding.py \
  --model-id Qwen/Qwen2.5-3B \
  --adapter-path checkpoints/exit_adapters_qwen_mlp.pt \
  --draft-depth 24 \
  --gamma 2 \
  --max-tokens 40 \
  --dtype float16 \
  --device cuda \
  2>&1 | tee results/raw_runs/gamma2_smoke/stdout.txt
```

#### Tuned-lens training

```bash
%cd /kaggle/working/kv-cache-continuity
!mkdir -p /kaggle/working/kv-cache-continuity/checkpoints \
  /kaggle/working/kv-cache-continuity/results/raw_runs/tuned_lens_training
!python -u experiments/train_tuned_lens.py \
  --model-id Qwen/Qwen2.5-3B \
  --target-depths 14,18,22,24,28 \
  --adapter-type lora \
  --rank 64 \
  --steps 400 \
  --batch-size 2 \
  --seq-len 128 \
  --lr 1e-3 \
  --distill-temp 2.0 \
  --output-path /kaggle/working/kv-cache-continuity/checkpoints/exit_adapters_qwen_tuned_lens.pt \
  --device cuda \
  2>&1 | tee /kaggle/working/kv-cache-continuity/results/raw_runs/tuned_lens_training/stdout.txt
```

The checkpoint and log paths above are explicitly under `/kaggle/working`,
which is writable. Do not run this cell from `/kaggle/input`, because Kaggle
mounts input datasets read-only. If training reached step 400 but failed with
`Read-only file system`, rerun the cell after confirming that the repository is
under `/kaggle/working`; the adapter is saved only after the final optimization
step.

#### Baseline MLP sweep

```bash
%cd /kaggle/working/kv-cache-continuity
!python experiments/sweep_speculative_depths.py \
  --model-id Qwen/Qwen2.5-3B \
  --adapter-path checkpoints/exit_adapters_qwen_mlp.pt \
  --depths 14,18,22,24,28,30 \
  --gamma 1 \
  --num-prompts 12 \
  --max-tokens 30 \
  --dtype float16 \
  --device cuda \
  --output-dir results/canonical_sweeps/qwen2.5_3b
```

#### Tuned-lens sweep

```bash
%cd /kaggle/working/kv-cache-continuity
!python experiments/sweep_speculative_depths.py \
  --model-id Qwen/Qwen2.5-3B \
  --adapter-path checkpoints/exit_adapters_qwen_tuned_lens.pt \
  --depths 14,18,22,24,28,30 \
  --gamma 1 \
  --num-prompts 12 \
  --max-tokens 30 \
  --dtype float16 \
  --device cuda \
  --output-dir results/canonical_sweeps/qwen2.5_3b
```

#### Tuned-lens 60-prompt benchmark

```bash
%cd /kaggle/working/kv-cache-continuity
!python experiments/benchmark_large_scale.py \
  --model-id Qwen/Qwen2.5-3B \
  --adapter-path checkpoints/exit_adapters_qwen_tuned_lens.pt \
  --draft-depth 24 \
  --gamma 1 \
  --num-prompts-per-domain 15 \
  --max-tokens 30 \
  --dtype float16 \
  --device cuda \
  --output-json results/canonical_sweeps/qwen2.5_3b/benchmark_tuned_lens_results.json
```

## 11. Download Kaggle results

Before the Kaggle session ends, create one archive:

```bash
%cd /kaggle/working/kv-cache-continuity
!tar -czf /kaggle/working/kv_continuity_next_results.tar.gz \
  checkpoints/exit_adapters_qwen_tuned_lens.pt \
  results/canonical_sweeps/qwen2.5_3b \
  results/raw_runs
```

Download `kv_continuity_next_results.tar.gz` from the Kaggle file browser. Extract
it into the repository root and verify the files before using any number in the
manuscript.

## 12. Final validation checklist

- [ ] `gamma=2` completed without NaN, shape, or CUDA errors.
- [ ] Tuned-lens checkpoint is present and non-empty.
- [ ] MLP and tuned-lens sweeps use identical benchmark settings.
- [ ] Both sweep CSV/PDF outputs were preserved under different names.
- [ ] The tuned-lens 60-prompt JSON contains all four domains and 60 prompts.
- [ ] Speedup, acceptance, and exact-match metrics were copied from JSON/logs,
      not manually retyped.
- [ ] No claim of `>1.0x` speedup is made unless the measured benchmark supports
      it under the same settings.
- [ ] The existing baseline discrepancy (`0.728x` in one summary versus
      `0.748x` in the canonical JSON) is resolved before manuscript submission.

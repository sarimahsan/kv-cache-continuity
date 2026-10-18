# Experiment 2: Tuned-Lens Adapter Training

**Status:** Complete  
**Run date:** 2026-10-15  
**Run environment:** Kaggle GPU  
**Experiment stage:** Next-experiments runbook, Experiment 2  
**Purpose:** Train a low-rank tuned-lens readout adapter for intermediate
hidden-state calibration.

## Configuration

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-3B` |
| Device | CUDA GPU |
| Adapter type | LoRA |
| Target depths | `14, 18, 22, 24, 28` |
| Rank | `64` |
| Training steps | `400` |
| Batch size | `2` |
| Sequence length | `128` |
| Learning rate | `1e-3` |
| Distillation temperature | `2.0` |
| Adapter parameters | `1,331,200` (`1.33M`) |

## Outcome

Training completed all `400/400` optimization steps in approximately `270.2`
seconds. The checkpoint was saved successfully to the Kaggle working
directory:

```text
/kaggle/working/kv-cache-continuity/checkpoints/exit_adapters_qwen_tuned_lens.pt
```

The downloaded local copy is stored at:

```text
checkpoints/exit_adapters_qwen_tuned_lens.pt
```

The checkpoint has a non-zero size and is ready for the tuned-lens depth sweep.

## Training log

| Step | Total loss | Elapsed time |
|---:|---:|---:|
| 50 | 1056.0000 | 31.8 s |
| 100 | 828.0000 | 66.7 s |
| 150 | 792.0000 | 100.4 s |
| 200 | 556.0000 | 134.2 s |
| 250 | 600.0000 | 168.4 s |
| 300 | 568.0000 | 202.3 s |
| 350 | 382.0000 | 236.2 s |
| 400 | 376.0000 | 270.2 s |

## Warnings

- Hugging Face reported unauthenticated requests. This did not prevent model
  download or training.
- Transformers reported that `torch_dtype` is deprecated in favor of `dtype`.
  This did not affect the run.

## Reproducibility notes

- The training script uses a fixed six-text synthetic/domain sample list and
  cycles through it during optimization.
- The checkpoint was downloaded from Kaggle after the successful run.
- This experiment measures adapter training completion only. It does not
  establish acceptance, speedup, or lossless equivalence. Those metrics require
  the subsequent depth sweep and benchmark.

## Source console summary

```text
FAST TUNED-LENS READOUT HEAD CALIBRATION
Model ID: Qwen/Qwen2.5-3B | Depths: [14, 18, 22, 24, 28] | Type: lora | Rank: 64
Steps: 400 | Batch Size: 2 | Seq Len: 128
Initialized ExitCalibrationModule with 1,331,200 parameters (1.33M params).
Step [  50/400] | Total Loss: 1056.0000 | Time: 31.8s
Step [ 100/400] | Total Loss: 828.0000 | Time: 66.7s
Step [ 150/400] | Total Loss: 792.0000 | Time: 100.4s
Step [ 200/400] | Total Loss: 556.0000 | Time: 134.2s
Step [ 250/400] | Total Loss: 600.0000 | Time: 168.4s
Step [ 300/400] | Total Loss: 568.0000 | Time: 202.3s
Step [ 350/400] | Total Loss: 382.0000 | Time: 236.2s
Step [ 400/400] | Total Loss: 376.0000 | Time: 270.2s
Successfully trained and saved tuned-lens calibration module.
```

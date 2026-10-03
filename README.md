# TextVQA Qwen3-VL PEFT Fine-Tuning (Parameter-Golf)

This repository contains the parameter-efficient fine-tuning (PEFT) implementation for `Qwen/Qwen3-VL-2B-Instruct` on TextVQA under strict compute, memory, and latency constraints.

Our champion solution achieves **73.790% Exact Match** on the full TextVQA validation set (5,000 samples) with a 3-seed mean of **73.461%**, significantly surpassing the base model zero-shot (69.84%) and the initial baseline (70.68%), while maintaining **0.998x** latency ratio and zero FLOPs overhead.

## Performance Overview

| Model / Method | Exact Match (val) ↑ | 3-Seed Mean ↑ | Latency Ratio ↓ | Train Time ↓ | Adapter Size ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-2B-Instruct (Zero-Shot) | 69.838% | - | 1.000x | 0s | 0 MB |
| Initial Baseline (Attention LoRA) | 70.680% | 70.680% | 1.000x | 3580s | 17.5 MB |
| **Our Champion Model (Seed 1)** | **73.790%** | **73.461%** | **0.998x** | **3496s** | **69.56 MB** |

### Per-Seed Results (5,000 samples each)

All three seeds are evaluated with the identical **Raw + EMA (50/50 Intra-Basin Soup)** methodology:

| Seed | Checkpoint / Fusion | Exact Match (%) ↑ | Stderr (CLT) | Output JSON |
| :--- | :--- | :---: | :---: | :--- |
| **Seed 1** | **Raw + EMA (50/50 Intra-Basin Soup)** | **73.790%** | ±0.00587 | `results/textvqa/seed1_champ_results.json` |
| **Seed 2** | **Raw + EMA (50/50 Intra-Basin Soup)** | **72.958%** | ±0.00593 | `results/textvqa/seed2_champ_results.json` |
| **Seed 3** | **Raw + EMA (50/50 Intra-Basin Soup)** | **73.634%** | ±0.00587 | `results/textvqa/seed3_champ_results.json` |
| **Mean** | **3-Seed Statistical Mean** | **73.461%** | **±0.442%** | - |

> Full evaluation logs and official lmms-eval output JSONs are recorded in `results/textvqa/`.

---

## Core Methodological Highlights

1. **Hardware Throughput Maximization**: Disabled gradient checkpointing under 11GB VRAM budget, boosting training throughput by +55.8% (from 0.254 to 0.396 steps/sec) and yielding 1,427 effective optimization steps within 3,600 seconds.
2. **LoRA+ Optimization Dynamics**: Decoupled learning rates between projection matrices ($\eta_B / \eta_A = 4.0$), eliminating feature scaling lag and accelerating loss convergence.
3. **Vision Backbone Selective Fine-Tuning**: Selectively unfroze high-level semantic blocks (ViT Blocks 21~23) while keeping the cross-modal Merger frozen, significantly improving fine-grained text localization.
4. **Intra-Basin Model Soup & Online EMA**: Implemented online exponential moving average tracking (decay 0.99) coupled with post-training 50/50 convex combination in weight space, eliminating high-frequency batch noise under strong local convexity ($\cos \theta = 0.9945$).

For in-depth theoretical motivations, mathematical derivations, and 8 systematically vetoed hypotheses, please refer to:
- [`REPORT.md`](REPORT.md): Publication-grade academic research report.
- [`EXPERIMENT_DETAILS.md`](EXPERIMENT_DETAILS.md): Supplementary material with hardware specs, full 59-experiment registry, and statistical variance proofs.

---

## Deliverables & Repository Structure

```text
parameter-golf/
├── assets/                                   # Visual assets for reports
│   ├── diagrams/                             # Draw.io architectural SVG diagrams
│   └── figures/                              # Python empirical experimental PNG curves
├── configs/
│   ├── vlm_textvqa_lora.yaml                 # Champion training configuration
│   ├── exp02_nogc.yaml ~ exp11_*.yaml        # Systematic ablation configs
├── results/
│   └── textvqa/
│       ├── seed1_champ_results.json          # Official 5k evaluation result (73.790%)
│       ├── seed2_champ_results.json          # Official 5k evaluation result (72.958%)
│       └── seed3_champ_results.json          # Official 5k evaluation result (73.634%)
├── weights/
│   ├── adapter_config.json                   # PEFT configuration
│   ├── adapter_model.safetensors             # Champion LoRA weights (69.56 MB)
│   ├── soup_manifest.json                    # Convex interpolation manifest
│   ├── training_config.json                  # Training arguments dump
│   └── README.md                             # Weight specifications
├── eval_qwen.sh                              # lmms-eval evaluation entrypoint
├── get_peft_loraplus.py                      # LoRA+ optimizer parameter grouping
├── merge_lora.py                             # Lossless LoRA weight merger
├── model_soup.py                             # Weight-space model soup utility
├── patch_optimizer.py                        # Optimizer patch helper
├── prepare_textvqa.py                        # Dataset prompt & cache preparation
├── run_merge_lora.sh                         # Merge execution script
├── run_prepare.sh                            # Data preparation script
├── run_train.sh                              # Single & multi-GPU training entrypoint
├── train_textvqa_qwen3vl.py                  # Core training script
├── EXPERIMENT_DETAILS.md                     # Supplementary material & experiment registry
├── README.md                                 # Project documentation
└── REPORT.md                                 # Full empirical research report
```

---

## Quick Start

### 1. Direct Evaluation of Submitted Champion Weights (Fastest)

To evaluate the submitted champion weights (73.790%) without retraining:

```bash
# 1. Merge submitted lightweight adapter (weights/) into base model
BASE_MODEL=/storage/yiguang/all_models/Qwen3-VL-2B-Instruct \
ADAPTER=./weights \
MERGED_MODEL=./outputs/champion_merged \
bash run_merge_lora.sh

# 2. Evaluate on full TextVQA validation set (5,000 samples)
MODEL_PATH=./outputs/champion_merged bash eval_qwen.sh
```

---

### 2. End-to-End Retraining & Reproduction

#### Step 1: Prepare Dataset

```bash
SEED=1 bash run_prepare.sh
```

#### Step 2: Train Model (Finished within 3600 seconds)

```bash
# Train on single GPU (or multi-GPU)
CUDA_VISIBLE_DEVICES=0 SEED=1 bash run_train.sh
```

*Note: Training automatically saves the raw checkpoint (`final_raw/`), the EMA checkpoint (`final_ema/`), and the auto-fused champion adapter (`final/`) at the end of the run.*

#### Step 3: Merge LoRA Adapter

```bash
SEED=1 \
BASE_MODEL=/storage/yiguang/all_models/Qwen3-VL-2B-Instruct \
ADAPTER=./outputs/textvqa_qwen3vl_lora_seed1/final \
MERGED_MODEL=./outputs/textvqa_qwen3vl_lora_seed1/merged \
bash run_merge_lora.sh
```

#### Step 4: Evaluate

```bash
MODEL_PATH=./outputs/textvqa_qwen3vl_lora_seed1/merged bash eval_qwen.sh
```

To run across all 3 seeds:

```bash
for seed in 1 2 3; do
  SEED=$seed bash run_prepare.sh
  SEED=$seed bash run_train.sh
  SEED=$seed bash run_merge_lora.sh
  MODEL_PATH=./outputs/textvqa_qwen3vl_lora_seed${seed}/merged bash eval_qwen.sh
done
```

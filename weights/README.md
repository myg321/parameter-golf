# Lightweight PEFT Weights (Seed 1 Champion Model)

This folder contains the lightweight LoRA PEFT adapter weights for the champion submission model achieving **73.790% Exact Match** on the full TextVQA validation set (5,000 samples).

## Specifications

- **Base Model**: `Qwen/Qwen3-VL-2B-Instruct`
- **Method**: Vision-Language Targeted LoRA+ with Online EMA & Intra-basin Model Soup (50/50 convex interpolation)
- **Target Modules**: Language attention & MLP (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) + ViT deep blocks 21~23 (`attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2`)
- **LoRA Hyperparameters**: $r=16, \alpha=32$, LoRA+ ratio $\lambda=4$
- **Weight File**: `adapter_model.safetensors` (~69.6 MB)
- **Adapter Config**: `adapter_config.json`

## Evaluation

To evaluate directly with the base model:

```bash
# Merge adapter into base model
BASE_MODEL=/storage/yiguang/all_models/Qwen3-VL-2B-Instruct \
ADAPTER=./weights \
MERGED_MODEL=./outputs/champion_merged \
bash run_merge_lora.sh

# Run evaluation on TextVQA validation set
MODEL_PATH=./outputs/champion_merged bash eval_qwen.sh
```

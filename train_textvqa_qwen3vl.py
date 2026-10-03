#!/usr/bin/env python
import argparse
import json
import os
import time

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed


DEFAULT_CONFIG = "configs/vlm_textvqa_lora.yaml"
SYSTEM_PROMPT = "You are a helpful assistant."


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(os.getenv("SEED", cfg.get("seed", 1)))
    cfg["seed"] = seed
    cfg["output_dir"] = os.getenv("OUTPUT_DIR", cfg["output_dir"]).format(seed=seed)
    cfg["prepared_data_dir"] = os.getenv("PREPARED_DATA_DIR", cfg["prepared_data_dir"]).format(seed=seed)
    return cfg


def load_prepared_dataset(cfg):
    if not os.path.isdir(cfg["prepared_data_dir"]):
        raise FileNotFoundError(
            f"Prepared dataset not found: {cfg['prepared_data_dir']}. "
            f"Run `python prepare_textvqa.py --config {DEFAULT_CONFIG}` first."
        )
    ds = load_from_disk(cfg["prepared_data_dir"])
    print(f"[INFO] Loaded {len(ds)} prepared TextVQA samples from {cfg['prepared_data_dir']}")
    return ds


class TimeLimitCallback(TrainerCallback):
    def __init__(self, max_seconds):
        self.max_seconds = max_seconds
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.max_seconds > 0 and time.time() - self.start_time > self.max_seconds:
            print(f"[TIMEOUT] Reached {self.max_seconds / 60:.1f} minute training budget")
            control.should_training_stop = True
        return control


class EMACallback(TrainerCallback):
    def __init__(self, decay=0.99, start_step=800):
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.shadow = {}
        self.step_count = 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step >= self.start_step and model is not None:
            self.step_count += 1
            d = min(self.decay, (1.0 + self.step_count) / (10.0 + self.step_count))
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if name not in self.shadow:
                            self.shadow[name] = param.data.detach().clone().cpu()
                        else:
                            self.shadow[name].mul_(d).add_(param.data.detach().cpu(), alpha=1.0 - d)
        return control

    def apply_ema(self, model):
        if not self.shadow or model is None:
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name].to(param.device))



class LoRATrainer(Trainer):
    def __init__(self, *args, loraplus_lr_ratio=None, eos_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.eos_loss_weight = float(eos_loss_weight)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.eos_loss_weight <= 1.0:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        weights = torch.ones_like(shift_labels.view(-1), dtype=torch.float32)
        weights[shift_labels.view(-1) == 151645] = self.eos_loss_weight
        valid_mask = (shift_labels.view(-1) != -100)
        weighted_loss = (loss * weights * valid_mask).sum() / (weights * valid_mask).sum().clamp(min=1.0)
        return (weighted_loss, outputs) if return_outputs else weighted_loss

    def create_optimizer(self):
        if self.loraplus_lr_ratio is not None and float(self.loraplus_lr_ratio) > 1.0:
            if self.optimizer is None:
                self.optimizer = create_loraplus_optimizer(
                    model=self.model,
                    optimizer_cls=torch.optim.AdamW,
                    lr=self.args.learning_rate,
                    loraplus_lr_ratio=float(self.loraplus_lr_ratio),
                    loraplus_weight_decay=self.args.weight_decay,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
                print(f"[INFO] LoRA+ optimizer created with ratio={self.loraplus_lr_ratio}, base_lr={self.args.learning_rate}")
                for idx, group in enumerate(self.optimizer.param_groups):
                    print(f"  Group {idx}: lr={group['lr']}, weight_decay={group.get('weight_decay', 0.0)}, num_params={len(group['params'])}")
            return self.optimizer
        return super().create_optimizer()


class TextVQADataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, processor, cfg):
        self.ds = hf_ds
        self.processor = processor
        self.cfg = cfg

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        answer = item["target_answer"]
        user_text = item["user_text"]

        prompt_conv = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
        ]
        full_conv = prompt_conv + [{"role": "assistant", "content": answer}]

        prompt_text = self.processor.apply_chat_template(prompt_conv, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_conv, tokenize=False, add_generation_prompt=False)

        common_kwargs = dict(
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=int(self.cfg.get("max_seq_length", 1024)),
        )
        prompt_batch = self.processor(text=prompt_text, **common_kwargs)
        full_batch = self.processor(text=full_text, **common_kwargs)

        input_ids = full_batch["input_ids"][0]
        attention_mask = full_batch["attention_mask"][0]
        labels = input_ids.clone()
        labels[: min(prompt_batch["input_ids"].shape[1], labels.shape[0])] = -100

        result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if "pixel_values" in full_batch:
            result["pixel_values"] = full_batch["pixel_values"]
        if "image_grid_thw" in full_batch:
            result["image_grid_thw"] = full_batch["image_grid_thw"]
        return result


def collate_fn(examples, processor):
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    batch = {
        "input_ids": pad_sequence([ex["input_ids"] for ex in examples], batch_first=True, padding_value=pad_id),
        "attention_mask": pad_sequence([ex["attention_mask"] for ex in examples], batch_first=True, padding_value=0),
        "labels": pad_sequence([ex["labels"] for ex in examples], batch_first=True, padding_value=-100),
    }

    if "pixel_values" in examples[0]:
        batch["pixel_values"] = torch.cat([ex["pixel_values"] for ex in examples], dim=0)
    if "image_grid_thw" in examples[0]:
        batch["image_grid_thw"] = torch.cat([ex["image_grid_thw"] for ex in examples], dim=0)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    processor = AutoProcessor.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True,
        max_pixels=int(cfg["max_pixels"]),
        min_pixels=int(cfg["min_pixels"]),
    )
    model = AutoModelForVision2Seq.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if hasattr(model, "visual"):
        for param in model.visual.parameters():
            param.requires_grad = False

    lora_kwargs = {
        "r": int(cfg["lora_r"]),
        "lora_alpha": int(cfg["lora_alpha"]),
        "lora_dropout": float(cfg["lora_dropout"]),
        "target_modules": cfg["target_modules"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if "rank_pattern" in cfg and cfg["rank_pattern"]:
        lora_kwargs["rank_pattern"] = cfg["rank_pattern"]
    if "alpha_pattern" in cfg and cfg["alpha_pattern"]:
        lora_kwargs["alpha_pattern"] = cfg["alpha_pattern"]

    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    use_gc = bool(cfg.get("gradient_checkpointing", True))
    if use_gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    raw_ds = load_prepared_dataset(cfg)
    train_ds = TextVQADataset(raw_ds, processor, cfg)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        max_steps=int(cfg["max_steps"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        weight_decay=float(cfg["weight_decay"]),
        adam_beta1=float(cfg.get("adam_beta1", 0.9)),
        adam_beta2=float(cfg.get("adam_beta2", 0.999)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(cfg["logging_steps"]),
        save_strategy="no",
        fp16=True,
        bf16=False,
        dataloader_num_workers=int(cfg["dataloader_num_workers"]),
        remove_unused_columns=False,
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        optim="adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    callbacks = [TimeLimitCallback(int(cfg.get("max_train_seconds", 0)))]
    ema_callback = None
    if bool(cfg.get("use_ema", False)):
        ema_callback = EMACallback(
            decay=float(cfg.get("ema_decay", 0.99)),
            start_step=int(cfg.get("ema_start_step", 800)),
        )
        callbacks.append(ema_callback)
        print(f"[INFO] Attached EMACallback with decay={cfg.get('ema_decay', 0.99)}, start_step={cfg.get('ema_start_step', 800)}")

    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=lambda examples: collate_fn(examples, processor),
        callbacks=callbacks,
        loraplus_lr_ratio=cfg.get("loraplus_lr_ratio", None),
        eos_loss_weight=cfg.get("eos_loss_weight", 1.0),
    )
    trainer.train()

    if ema_callback is not None and ema_callback.shadow:
        final_raw_dir = os.path.join(cfg["output_dir"], "final_raw")
        os.makedirs(final_raw_dir, exist_ok=True)
        trainer.save_model(final_raw_dir)
        processor.save_pretrained(final_raw_dir)
        with open(os.path.join(final_raw_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        print(f"[INFO] Successfully saved RAW adapter model to {final_raw_dir}")

        raw_params = {
            name: param.data.detach().clone()
            for name, param in model.named_parameters()
            if name in ema_callback.shadow
        }

        final_ema_dir = os.path.join(cfg["output_dir"], "final_ema")
        os.makedirs(final_ema_dir, exist_ok=True)
        ema_callback.apply_ema(model)
        trainer.save_model(final_ema_dir)
        processor.save_pretrained(final_ema_dir)
        with open(os.path.join(final_ema_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        print(f"[INFO] Successfully saved EMA adapter model to {final_ema_dir}")

        final_dir = os.path.join(cfg["output_dir"], "final")
        os.makedirs(final_dir, exist_ok=True)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in raw_params:
                    param.data.mul_(0.5).add_(raw_params[name], alpha=0.5)
        trainer.save_model(final_dir)
        processor.save_pretrained(final_dir)
        with open(os.path.join(final_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        print(f"[INFO] Successfully saved 50/50 Raw+EMA Auto-Fused champion adapter model to {final_dir}")
    else:
        final_dir = os.path.join(cfg["output_dir"], "final")
        os.makedirs(final_dir, exist_ok=True)
        trainer.save_model(final_dir)
        processor.save_pretrained(final_dir)
        with open(os.path.join(final_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        print(f"[INFO] Successfully saved adapter model to {final_dir}")


if __name__ == "__main__":
    main()

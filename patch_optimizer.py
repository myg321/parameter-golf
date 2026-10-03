import torch
def custom_create_loraplus_optimizer(model, optimizer_cls, lr, loraplus_lr_ratio, vit_lr_ratio, **kwargs):
    from peft.optimizers import get_parameter_names, ALL_LAYERNORM_LAYERS
    from operator import attrgetter
    from torch.nn import Embedding
    
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupA_llm": {}, "groupB_llm": {}, "groupB_no_decay_llm": {},
        "groupA_vit": {}, "groupB_vit": {}, "groupB_no_decay_vit": {},
        "embedding": {},
    }
    
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        module = attrgetter(name)(model)
        is_vit = "visual" in name
        
        if isinstance(module, Embedding):
            param_groups["embedding"][name] = param
        elif "lora_B" in name or param.ndim == 1:
            if name in decay_parameters:
                if is_vit: param_groups["groupB_vit"][name] = param
                else: param_groups["groupB_llm"][name] = param
            else:
                if is_vit: param_groups["groupB_no_decay_vit"][name] = param
                else: param_groups["groupB_no_decay_llm"][name] = param
        else:
            if is_vit: param_groups["groupA_vit"][name] = param
            else: param_groups["groupA_llm"][name] = param

    loraplus_weight_decay = kwargs.pop("loraplus_weight_decay", 0.0)
    loraplus_lr_embedding = kwargs.pop("loraplus_lr_embedding", 1e-6)

    optimizer_grouped_parameters = [
        {"params": list(param_groups["groupA_llm"].values()), "weight_decay": loraplus_weight_decay, "lr": lr},
        {"params": list(param_groups["groupA_vit"].values()), "weight_decay": loraplus_weight_decay, "lr": lr * vit_lr_ratio},
        {"params": list(param_groups["groupB_llm"].values()), "weight_decay": loraplus_weight_decay, "lr": lr * loraplus_lr_ratio},
        {"params": list(param_groups["groupB_vit"].values()), "weight_decay": loraplus_weight_decay, "lr": lr * vit_lr_ratio * loraplus_lr_ratio},
        {"params": list(param_groups["groupB_no_decay_llm"].values()), "weight_decay": 0.0, "lr": lr * loraplus_lr_ratio},
        {"params": list(param_groups["groupB_no_decay_vit"].values()), "weight_decay": 0.0, "lr": lr * vit_lr_ratio * loraplus_lr_ratio},
        {"params": list(param_groups["embedding"].values()), "weight_decay": loraplus_weight_decay, "lr": loraplus_lr_embedding},
    ]
    # filter out empty groups
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]
    return optimizer_cls(optimizer_grouped_parameters, **kwargs)

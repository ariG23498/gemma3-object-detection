# Optional – comment this out if you are not planinng to use unsloth
try:
    from unsloth import FastModel
except ImportError:
    FastModel = None  # will be checked at runtime
# FastModel = None

import logging
import wandb
from functools import partial

import torch
from torch.amp import autocast, GradScaler
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers import BitsAndBytesConfig

from utils.config import Configuration
from utils.utilities import train_collate_function, train_collate_function_unsloth, save_best_model, push_to_hub
from peft import get_peft_config, get_peft_model, LoraConfig
import albumentations as A

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


augmentations = A.Compose([
    A.Resize(height=896, width=896),
    # A.HorizontalFlip(p=0.5), # does this handle flipping box coordinates? 
    A.ColorJitter(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], filter_invalid_bboxes=True))

def get_dataloader_unsloth(tokenizer, args, dtype, split="train"):
    logger.info("Fetching the dataset")
    train_dataset = load_dataset(args.dataset_id, split=split)  # or cfg.dataset_id
    train_collate_fn = partial(
        train_collate_function_unsloth,
        tokenizer=tokenizer,         # <- Use the Unsloth tokenizer
        dtype=dtype,
        transform=augmentations
    )

    logger.info("Building data loader")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
    )
    return train_dataloader

def get_dataloader(processor, args, dtype, tokenizer=None, split="train"):
    logger.info("Fetching the dataset")
    train_dataset = load_dataset(cfg.dataset_id, split=split)
    train_collate_fn = partial(
        train_collate_function, processor=processor, dtype=dtype, transform=augmentations
    )

    logger.info("Building data loader")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
        pin_memory=True,
    )
    return train_dataloader

def step(model, batch, device, use_fp16, optimizer=None, scaler=None):
    data = batch.to(device)
    if use_fp16:
        with autocast(device_type=device):
            loss = model(**data).loss
    else:
        loss = model(**data).loss
    if optimizer:
        optimizer.zero_grad()
        if use_fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    return loss.item()

def validate_all(model, val_loader, device, use_fp16, val_bathes=None):
    model.eval()
    with torch.no_grad():
        if val_bathes:
            ## TODO: This logic is Temp and should be removed in final clean up
            n_batches = 10
            losses = []
            for i, batch in enumerate(val_loader):
                if i >= n_batches:
                    break
                losses.append(step(model, batch, device, use_fp16))
        else:
            losses = [step(model, batch, device, use_fp16) for batch in val_loader]
    model.train()
    return sum(losses) / len(losses) if len(losses)> 0 else 0

def train_model(model, optimizer, cfg, train_loader, val_loader=None, val_every=5, push_hub=False):
    use_fp16 = cfg.dtype in [torch.float16, torch.bfloat16]
    scaler = GradScaler() if use_fp16 else None
    global_step, best_val_loss = 0, float("inf")

    for epoch in range(cfg.epochs):
        for idx, batch in enumerate(train_loader):
            loss = step(model, batch, cfg.device, use_fp16, optimizer, scaler)
            if global_step % 1 == 0:
                logger.info(f"Epoch:{epoch} Step:{global_step} Loss:{loss:.4f}")
                wandb.log({"train/loss": loss, "epoch": epoch}, step=global_step)
            if val_loader and global_step % val_every == 0:
                val_loss = validate_all(model, val_loader, cfg.device, use_fp16)
                logger.info(f"Step:{global_step} Val Loss:{val_loss:.4f}")
                wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)
            global_step += 1

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_model(model, cfg, tokenizer, cfg.finetune_method in {"lora", "qlora"}, logger)

    return model


def load_model(cfg:Configuration):

    lcfg = cfg.lora
    tokenizer = None

    if cfg.use_unsloth and FastModel is not None:

        model, tokenizer = FastModel.from_pretrained(
            model_name = "unsloth/gemma-3-4b-it",
            max_seq_length = 2048, # Choose any for long context!
            load_in_4bit = True,  # 4 bit quantization to reduce memory
            load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
            full_finetuning = False, # [NEW!] We have full finetuning now!
            # token = "hf_...", # use one if using gated models
        )

        if cfg.finetune_method in {"lora", "qlora"}:
            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers     = True, # Turn off for just text!
                finetune_language_layers   = True,  # Should leave on!
                finetune_attention_modules = True,  # Attention good for GRPO
                finetune_mlp_modules       = True,  # SHould leave on always!

                r=lcfg.r, # Larger = higher accuracy, but might overfit
                lora_alpha=lcfg.alpha, # Recommended alpha == r at least
                lora_dropout=lcfg.dropout,
                bias = "none",
                random_state = 3407,
            )

            model.print_trainable_parameters()


    else:
        quant_args = {}
        # Enable quantization only for QLoRA or if specifically requested for LoRA
        if cfg.finetune_method in {"lora", "qlora"}:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=cfg.dtype,
            )
            quant_args = {"quantization_config": bnb_config, "device_map": "auto"}

        model = Gemma3ForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.dtype,
            attn_implementation="eager",
            **quant_args,
        )

        if cfg.finetune_method in {"lora", "qlora"}:
            for n, p in model.named_parameters():
                p.requires_grad = False

            lora_cfg = LoraConfig(
                r=lcfg.r,
                lora_alpha=lcfg.alpha,
                target_modules=lcfg.target_modules,
                lora_dropout=lcfg.dropout,
                bias="none",
            )
            
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
            torch.cuda.empty_cache()

        elif cfg.finetune_method == "FFT":
            # Only unfreeze requested model parts (e.g. multi_modal_projector)
            for n, p in model.named_parameters():
                p.requires_grad = any(part in n for part in cfg.mm_tunable_parts)
                print(f"{n} will be finetuned")
        else:
            raise ValueError(f"Unknown finetune_method: {cfg.finetune_method}")
    
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"{n} will be finetuned")

    return model, tokenizer


if __name__ == "__main__":
    # 1. Parse CLI + YAMLs into config
    cfg = Configuration.from_args()

    logger.info("Getting model & turning only attention parameters to trainable")
    model, tokenizer = load_model(cfg)

    if cfg.use_unsloth:
        train_dataloader = get_dataloader_unsloth(tokenizer=tokenizer, args=cfg, dtype=cfg.dtype)
        validation_dataloader = get_dataloader_unsloth(tokenizer=tokenizer, args=cfg, dtype=cfg.dtype, split="validation")
    else:
        processor = AutoProcessor.from_pretrained(cfg.model_id)
        train_dataloader = get_dataloader(processor=processor, args=cfg, dtype=cfg.dtype)
        validation_dataloader = get_dataloader(processor=processor, args=cfg, dtype=cfg.dtype, split="validation")
    
    model.train()
    model.to(cfg.device)

    # Credits to Sayak Paul for this beautiful expression
    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(params_to_train, lr=cfg.learning_rate)

    wandb.init(
        project=cfg.project_name,
        name=cfg.run_name if hasattr(cfg, "run_name") else None,
        config=vars(cfg),
    )

    train_model(model, optimizer, cfg, train_dataloader, validation_dataloader,val_every=10, push_hub=True)

    # TODO add flag to config (code tested and its working)
    # if push_hub:
    #     logger.info(f"Pushing to hub at: {cfg.checkpoint_id}")
    #     push_to_hub(model, cfg, tokenizer, cfg.finetune_method in {"lora", "qlora"})

    wandb.finish()
    logger.info("Train finished")

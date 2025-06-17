
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

FastModel = None
# Optional – comment below imports if you are not planinng to use unsloth
try: from unsloth import FastModel
except ImportError as e: logger.log(f"Unsloth import error : {e}")
except NotImplementedError as e: logger.log(f"Unsloth NotImplementedError error : {e}")
    

import wandb
from functools import partial
import torch
from torch.amp import autocast, GradScaler
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers import BitsAndBytesConfig

from utils.config import Configuration
from utils.utilities import train_collate_function, train_collate_function_unsloth
from utils.utilities import save_best_model, push_to_hub, load_saved_model
from peft import get_peft_config, get_peft_model, LoraConfig
import albumentations as A


augmentations = A.Compose([
    A.Resize(height=896, width=896),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], filter_invalid_bboxes=True))

# TODO: Delete this after testing get_dataloader() with is_unsloth=True flag
# def get_dataloader_unsloth(tokenizer, args, dtype, split="train"):
#     logger.info("Fetching the dataset")
#     train_dataset = load_dataset(args.dataset_id, split=split)  # or cfg.dataset_id
#     train_collate_fn = partial(
#         train_collate_function_unsloth,
#         tokenizer=tokenizer,         # <- Use the Unsloth tokenizer instead of processor
#         dtype=dtype,
#         transform=augmentations
#     )

#     logger.info("Building data loader")
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         collate_fn=train_collate_fn,
#         shuffle=True,
#     )
#     return train_dataloader

def get_dataloader(processor, args, dtype, split="train", is_unsloth=False):
    logger.info("Fetching the dataset")
    train_dataset = load_dataset(cfg.dataset_id, split=split)

    if is_unsloth:
        # <- Use the Unsloth tokenizer instead of processor
        train_collate_fn = partial(train_collate_function_unsloth,tokenizer=tokenizer,dtype=dtype,transform=augmentations)
    else:
        train_collate_fn = partial(train_collate_function, processor=processor, dtype=dtype, transform=augmentations)

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
    """
    Single batch process
    """
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

def validate_all(model, val_loader, cfg, use_fp16,val_batches=5):

    if cfg.use_unsloth and FastModel is not None:
        FastModel.for_inference(model) # Enable for inference!
    else:
        model.eval()

    with torch.no_grad():
        if val_batches:
            ## TODO: This logic is Temp and should be removed in final clean up
            n_batches = val_batches
            losses = []
            for i, batch in enumerate(val_loader):
                if i >= n_batches:
                    break
                losses.append(step(model, batch, cfg.device, use_fp16))
        else:
            losses = [step(model, batch, cfg.device, use_fp16) for batch in val_loader]
    model.train()
    return sum(losses) / len(losses) if len(losses)> 0 else 0

def train_model(model, optimizer, cfg:Configuration, train_loader, val_loader=None):
    use_fp16 = cfg.dtype in [torch.float16, torch.bfloat16]
    scaler = GradScaler() if use_fp16 else None
    global_step, best_val_loss = 0, float("inf")

    if cfg.use_unsloth and FastModel is not None:
        FastModel.for_training(model) # Enable for inference!
    else:
        model.train()
        model.to(cfg.device)

    for epoch in range(cfg.epochs):
        for idx, batch in enumerate(train_loader):
            loss = step(model, batch, cfg.device, use_fp16, optimizer, scaler)
            if global_step % 1 == 0:
                logger.info(f"Epoch:{epoch} Step:{global_step} Loss:{loss:.4f}")
                wandb.log({"train/loss": loss, "epoch": epoch}, step=global_step)
            if val_loader and global_step % cfg.validate_steps_freq == 0:
                val_loss = validate_all(model, val_loader, cfg, use_fp16, val_batches=5) # TODO, disable val_batches in final commit/run
                logger.info(f"Step:{global_step} Val Loss:{val_loss:.4f}")
                wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_model(model, cfg, tokenizer, cfg.finetune_method in {"lora", "qlora"}, logger)

            ## Model seem to converge before even first epoch finishes for LoRA. set max_step_to_train<=0 to disable this.
            if global_step>cfg.max_step_to_train-1 and cfg.max_step_to_train>0:
                break
            global_step += 1

    return model


def load_model(cfg:Configuration):

    lcfg = cfg.lora
    tokenizer = None

    if cfg.use_unsloth and FastModel is not None:

        # TODO: For LoRA and QLoRa change unsloth config accordigly, generally load_in_4bit, load_in_8bit will be False or LoRA
        model, tokenizer = FastModel.from_pretrained(
            model_name = "unsloth/gemma-3-4b-it",
            max_seq_length = 2048, # Choose any for long context!
            load_in_4bit = True,  # 4 bit quantization to reduce memory
            load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
            full_finetuning = False, # [NEW!] We have full finetuning now!
            # token = os.environ["HF_TOKEN"] # TODO: Handle this
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
        # Enable quantization only for QLoRA
        if cfg.finetune_method in {"qlora"}:
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

            lora_cfg = LoraConfig(
                r=lcfg.r,
                lora_alpha=lcfg.alpha,
                target_modules=lcfg.target_modules,
                lora_dropout=lcfg.dropout,
                bias="none",
            )
            
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
            torch.cuda.empty_cache() # TODO: Do I need this? Just want to make sure I have mem cleaned up before training starts.

        elif cfg.finetune_method == "FFT":
            # handled below before printing params
            pass
        else:
            raise ValueError(f"Unknown finetune_method: {cfg.finetune_method}")
    
    for n, p in model.named_parameters():
        if cfg.finetune_method == "FFT":
            p.requires_grad = any(part in n for part in cfg.mm_tunable_parts)
        if p.requires_grad:
            print(f"{n} will be finetuned")

    return model, tokenizer


if __name__ == "__main__":
    # 1. Parse CLI + YAMLs into config
    cfg = Configuration.from_args()  # config.yaml is overriden by CLI arguments

    # 2. Load model
    logger.info(f"Getting model for {cfg.finetune_method}")
    # loads model based on config. Unsloth, lora, qlora, FFT
    model, tokenizer = load_model(cfg)

    # 3. Get Data
    if cfg.use_unsloth:
        train_dataloader = get_dataloader(tokenizer=tokenizer, args=cfg, dtype=cfg.dtype,split="train",is_unsloth=True)
        validation_dataloader = get_dataloader(tokenizer=tokenizer, args=cfg, dtype=cfg.dtype, split="validation",is_unsloth=True)
    else:
        processor = AutoProcessor.from_pretrained(cfg.model_id)
        train_dataloader = get_dataloader(processor=processor, args=cfg, dtype=cfg.dtype, split="train")
        validation_dataloader = get_dataloader(processor=processor, args=cfg, dtype=cfg.dtype, split="validation")
    
    # Credits to Sayak Paul for this beautiful expression
    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(params_to_train, lr=cfg.learning_rate)

    # 5. Enable logging, need to login or set wanddb token in os.env
    wandb.init(
        project=cfg.wandb_project_name,
        name=cfg.run_name if hasattr(cfg, "run_name") else None,
        config=vars(cfg),
    )

    # 5. Actual train and validation, validation_dataloader=None to do just traing.
    train_model(model, optimizer, cfg, train_dataloader, validation_dataloader)

    # 6. Loading best model back
    model, tokenizer = load_saved_model(cfg, is_lora=cfg.finetune_method in {"lora", "qlora"}, device="cuda", logger=logger)
    logger.info(f"Pushing to hub at: {cfg.checkpoint_id}")
    if cfg.push_model_to_hub:
        push_to_hub(model, cfg, tokenizer, cfg.finetune_method in {"lora", "qlora"})

    # 7. Test?  # TODO
    
    
    # 8. Wrap up
    wandb.finish()
    logger.info("Train finished")

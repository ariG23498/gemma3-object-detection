import logging
import wandb
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import get_peft_model, prepare_model_for_kbit_training

from config import Configuration
from utils import train_collate_function

import albumentations as A

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define augmentations
augmentations = A.Compose([
    A.Resize(height=896, width=896),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], filter_invalid_bboxes=True))


def get_dataloader(processor, cfg):
    logger.info("Loading dataset")
    train_dataset = load_dataset(cfg.dataset_id, split="train")
    train_collate_fn = partial(
        train_collate_function, processor=processor, dtype=cfg.dtype, transform=augmentations
    )

    logger.info("Building DataLoader")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
    )
    return train_dataloader


def setup_model(cfg):
    logger.info("Loading model with QLoRA configuration")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.dtype,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=cfg.bnb_config if cfg.use_qlora else None,
        trust_remote_code=True,
    )

    if cfg.use_qlora:
        logger.info("Preparing model for QLoRA training")
        model = prepare_model_for_kbit_training(model)

        lora_config = cfg.lora_config
        lora_config.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        lora_config.task_type = "CAUSAL_LM"

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        logger.info("Traditional mode - training attention layers only")
        for name, param in model.named_parameters():
            param.requires_grad = "attn" in name

    return model


def train_model(model, optimizer, cfg, train_dataloader):
    logger.info("Starting training")
    global_step = 0

    for epoch in range(cfg.epochs):
        for idx, batch in enumerate(train_dataloader):
            # Move data to device
            batch = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            if idx % 100 == 0:
                logger.info(f"Epoch {epoch} | Step {idx} | Loss: {loss.item():.4f}")
                wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=global_step)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

    return model


if __name__ == "__main__":
    cfg = Configuration()

    # Initialize Weights & Biases
    wandb.init(
        project=cfg.project_name if hasattr(cfg, "project_name") else "gemma3-detection",
        name=cfg.run_name if hasattr(cfg, "run_name") else "run-qlora" if cfg.use_qlora else "run-traditional",
        config=vars(cfg),
    )

    # Preprocessing
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    train_dataloader = get_dataloader(processor, cfg)

    # Load model
    model = setup_model(cfg)
    model.train()

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)

    # Training
    trained_model = train_model(model, optimizer, cfg, train_dataloader)

    # Save
    logger.info("Saving model")
    if cfg.use_qlora:
        trained_model.save_pretrained(cfg.checkpoint_id)
        trained_model.push_to_hub(cfg.checkpoint_id)
    else:
        model.push_to_hub(cfg.checkpoint_id)

    processor.push_to_hub(cfg.checkpoint_id)

    wandb.finish()
    logger.info("Training complete")

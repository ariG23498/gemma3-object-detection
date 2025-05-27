import logging
import os
import wandb
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from config import Configuration
from utils import train_collate_function, test_collate_function

import albumentations as A

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


augmentations = A.Compose([
    A.Resize(height=896, width=896),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], filter_invalid_bboxes=True))


def get_dataloaders(processor):
    logger.info("Fetching the datasets")
    train_dataset = load_dataset(cfg.dataset_id, split="train")
    val_dataset = load_dataset(cfg.dataset_id, split="validation")
    
    train_collate_fn = partial(
        train_collate_function, processor=processor, dtype=cfg.dtype, transform=augmentations
    )
    val_collate_fn = partial(
        test_collate_function, processor=processor, dtype=cfg.dtype
    )

    logger.info("Building data loaders")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=val_collate_fn,
        shuffle=False,
    )
    return train_dataloader, val_dataloader


def evaluate_model(model, val_dataloader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            batch = batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_samples += batch["input_ids"].size(0)
    
    avg_loss = total_loss / total_samples
    model.train()
    return avg_loss


def train_model(model, optimizer, cfg, train_dataloader, val_dataloader):
    logger.info("Start training")
    global_step = 0
    best_val_loss = float('inf')
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(cfg.epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for idx, batch in enumerate(train_dataloader):
            batch = batch.to(model.device)
            outputs = model(**batch)
            loss = outputs.loss
            
            train_loss += loss.item() * batch["input_ids"].size(0)
            train_samples += batch["input_ids"].size(0)
            
            if idx % 100 == 0:
                logger.info(f"Epoch: {epoch} Iter: {idx} Loss: {loss.item():.4f}")
                wandb.log({
                    "train/step_loss": loss.item(),
                    "epoch": epoch,
                    "step": global_step
                })

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        avg_train_loss = train_loss / train_samples
        wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch})
        
        val_loss = evaluate_model(model, val_dataloader, cfg.device)
        wandb.log({"val/loss": val_loss, "epoch": epoch})
        logger.info(f"Epoch: {epoch} Train Loss: {avg_train_loss:.4f} Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
            logger.info(f"New best model saved at {checkpoint_path} with val loss {val_loss:.4f}")
            
            if epoch % cfg.save_every == 0:
                model.push_to_hub(cfg.checkpoint_id, commit_message=f"Epoch {epoch} - Val loss {val_loss:.4f}")
                processor.push_to_hub(cfg.checkpoint_id)
    
    return model


if __name__ == "__main__":
    cfg = Configuration()
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    train_dataloader, val_dataloader = get_dataloaders(processor)

    logger.info("Getting model & turning only attention parameters to trainable")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.dtype,
        device_map="cpu",
        attn_implementation="eager",
    )
    for name, param in model.named_parameters():
        if "attn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

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

    try:
        train_model(model, optimizer, cfg, train_dataloader, val_dataloader)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Push the final checkpoint to hub
        model.push_to_hub(cfg.checkpoint_id, commit_message="Final model")
        processor.push_to_hub(cfg.checkpoint_id)
        wandb.finish()
        logger.info("Training finished")

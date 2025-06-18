import logging
import wandb
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM

from config import Configuration
from utils import train_collate_function
import argparse
import albumentations as A

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_augmentations(cfg):
    if "SmolVLM" in cfg.model_id:
        resize_size = 512
    else:
        resize_size = 896

    augmentations = A.Compose([
        A.Resize(height=resize_size, width=resize_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], filter_invalid_bboxes=True))
    return augmentations



def get_dataloader(processor, cfg):
    logger.info("Fetching the dataset")
    train_dataset = load_dataset(cfg.dataset_id, split="train")
    train_collate_fn = partial(
        train_collate_function, processor=processor, device=cfg.device, transform=get_augmentations(cfg)
    )

    logger.info("Building data loader")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
    )
    return train_dataloader


def train_model(model, optimizer, cfg, train_dataloader):
    logger.info("Start training")
    global_step = 0
    for epoch in range(cfg.epochs):
        for idx, batch in enumerate(train_dataloader):
            outputs = model(**batch.to(model.device))
            loss = outputs.loss
            if idx % 100 == 0:
                logger.info(f"Epoch: {epoch} Iter: {idx}/{len(train_dataloader)} Loss: {loss.item():.4f}")
                wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=global_step)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
    return model


if __name__ == "__main__":
    cfg = Configuration()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, help='Model ID on Hugging Face Hub')
    parser.add_argument('--dataset_id', type=str, help='Dataset ID on Hugging Face Hub')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--checkpoint_id', type=str, help='Model repo to push to the Hub')

    args = parser.parse_args()

    if args.model_id: cfg.model_id = args.model_id
    if args.dataset_id: cfg.dataset_id = args.dataset_id
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.learning_rate: cfg.learning_rate = args.learning_rate
    if args.epochs: cfg.epochs = args.epochs
    if args.checkpoint_id: cfg.checkpoint_id = args.checkpoint_id

    processor = AutoProcessor.from_pretrained(cfg.model_id)
    train_dataloader = get_dataloader(processor=processor, cfg=cfg)

    logger.info("Getting model & turning only attention parameters to trainable")
    if "SmolVLM" in cfg.model_id:
        logger.info("Using AutoModelForVision2Seq")
        model = AutoModelForVision2Seq.from_pretrained(
            cfg.model_id,
            device_map="auto"
        )
    else:
        logger.info("Using AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.dtype,
            device_map="auto",
            _attn_implementation="eager",
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

    train_model(model, optimizer, cfg, train_dataloader)

    # Push the checkpoint to hub
    model.push_to_hub(cfg.checkpoint_id)
    processor.push_to_hub(cfg.checkpoint_id)

    wandb.finish()
    logger.info("Train finished")

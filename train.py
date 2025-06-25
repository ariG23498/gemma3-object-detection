import logging
import wandb
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM

from config import Configuration
from utils import train_collate_function, get_processor_with_new_tokens, get_model_with_resize_token_embeddings
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

def set_trainable_params(model, keywords):
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in keywords)


def run_training_phase(model, processor, cfg, train_dataloader, train_keys, phase_name="phase"):
    set_trainable_params(model, train_keys)
    model.train()
    model.to(cfg.device)

    params_to_train = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params_to_train, lr=cfg.learning_rate)

    wandb.init(
        project=cfg.project_name,
        name=f"{cfg.run_name}_{phase_name}" if hasattr(cfg, "run_name") else phase_name,
        config=vars(cfg),
    )

    train_model(model, optimizer, cfg, train_dataloader)
    wandb.finish()
c

if __name__ == "__main__":
    cfg = Configuration()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, help='Model ID on Hugging Face Hub')
    parser.add_argument('--dataset_id', type=str, help='Dataset ID on Hugging Face Hub')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--checkpoint_id', type=str, help='Model repo to push to the Hub')
    parser.add_argument('--include_loc_tokens', action='store_true', help='Include location tokens in the model.')

    args = parser.parse_args()

    if args.model_id: cfg.model_id = args.model_id
    if args.dataset_id: cfg.dataset_id = args.dataset_id
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.learning_rate: cfg.learning_rate = args.learning_rate
    if args.epochs: cfg.epochs = args.epochs
    if args.checkpoint_id: cfg.checkpoint_id = args.checkpoint_id

    processor = AutoProcessor.from_pretrained(cfg.model_id)
    if args.include_loc_tokens:
        logger.info("Adding location tokens to the tokenizer")
        processor = get_processor_with_new_tokens(processor)

    train_dataloader = get_dataloader(processor=processor, cfg=cfg)

    logger.info("Loading model")
    if "SmolVLM" in cfg.model_id:
        model = AutoModelForVision2Seq.from_pretrained(cfg.model_id, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=cfg.dtype, device_map="auto", _attn_implementation="eager")

    if args.include_loc_tokens:
        model = get_model_with_resize_token_embeddings(model, processor)

        logger.info("Stage 1: Training embed_tokens")
        run_training_phase(model, processor, cfg, train_dataloader, train_keys=["embed_tokens"], phase_name="embed_only")

        logger.info("Stage 2: Fine-tuning embed_tokens + attn")
        run_training_phase(model, processor, cfg, train_dataloader, train_keys=["embed_tokens", "attn"], phase_name="embed_attn")
    else:
        logger.info("Single-stage: Fine-tuning attn only")
        run_training_phase(model, processor, cfg, train_dataloader, train_keys=["attn"], phase_name="attn_only")

    model.push_to_hub(cfg.checkpoint_id)
    processor.push_to_hub(cfg.checkpoint_id)

    logger.info("Train finished")

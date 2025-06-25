import logging
import wandb
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from config import Configuration
from utils import train_collate_function, get_last_checkpoint_step, get_augmentations

import albumentations as A

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

augmentations = get_augmentations()

def get_dataloader(processor):
    logger.info("Fetching the dataset")
    train_dataset = load_dataset(cfg.dataset_id, split="train")
    train_collate_fn = partial(
        train_collate_function, processor=processor, dtype=cfg.dtype, transform=augmentations
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

    if cfg.resume:
        check_point_number = get_last_checkpoint_step(accelerator)
        global_step = check_point_number * cfg.checkpoint_interval +1
        starting_epoch = int(global_step/len(train_dataloader))
        skip_batch = global_step % len(train_dataloader)
        accelerator.project_configuration.iteration = check_point_number + 1
        skip_dataloader = accelerator.skip_first_batches(train_dataloader, skip_batch)
        accelerator.load_state()
    else:
        check_point_number = 0
        global_step = 0
        starting_epoch = 0
        skip_batch = 0
        accelerator.project_configuration.iteration = 0
        skip_dataloader = train_dataloader
        accelerator.save_state()
    
    for epoch in range(starting_epoch, cfg.epochs):
        for idx, batch in enumerate(skip_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            if (idx+skip_batch) % cfg.log_interval == 0:
                logger.info(f"Epoch: {epoch+1} Iter: {idx+skip_batch} Loss: {loss.item():.4f}")
                wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=global_step)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % cfg.checkpoint_interval == 0:
                accelerator.save_state()
        skip_dataloader = train_dataloader
        skip_batch = 0
    accelerator.end_training()
    return model


if __name__ == "__main__":
    cfg = Configuration()

    accelerator = Accelerator(
        project_config=ProjectConfiguration(
            project_dir=f"{cfg.project_dir}/{cfg.run_name}",
            logging_dir=f"{cfg.project_dir}/{cfg.run_name}/{cfg.log_dir}",
            automatic_checkpoint_naming = cfg.automatic_checkpoint_naming,
        ),
    )

    processor = AutoProcessor.from_pretrained(cfg.model_id)
    train_dataloader = get_dataloader(processor)

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

    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(params_to_train, lr=cfg.learning_rate)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    wandb.init(
        project=cfg.project_name if hasattr(cfg, "project_name") else "gemma3-object-detection",
        name=cfg.run_name if hasattr(cfg, "run_name") else "0",
        config=vars(cfg),
    )

    train_model(model, optimizer, cfg, train_dataloader)

    wandb.finish()
    logger.info("Train finished")

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.push_to_hub(cfg.checkpoint_id, safe_serialization=False)
    processor.push_to_hub(cfg.checkpoint_id)
    logger.info("Pushed Model to Hub")
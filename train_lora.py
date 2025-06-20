import logging
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftType
from config import Configuration
from utils import train_collate_function

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_dataloader(processor):
    logger.info("Fetching the dataset")
    train_dataset = load_dataset(cfg.dataset_id, split="train")
    train_collate_fn = partial(
        train_collate_function, processor=processor, dtype=cfg.dtype
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
    for epoch in range(cfg.epochs):
        for idx, batch in enumerate(train_dataloader):
            outputs = model(**batch.to(model.device))
            loss = outputs.loss
            if idx % 100 == 0:
                logger.info(f"Epoch: {epoch} Iter: {idx} Loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model


if __name__ == "__main__":
    cfg = Configuration()
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    train_dataloader = get_dataloader(processor)

    logger.info("Getting model & turning only attention parameters to trainable")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.dtype,
        device_map="cpu",
        attn_implementation="eager",
    )
    model.requires_grad_(False)
    lora_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        peft_type=PeftType.LORA,
    )

    lora_model = get_peft_model(model=model, peft_config=lora_config).to(cfg.device)
    lora_model.print_trainable_parameters()

    model.train()
    model.to(cfg.device)

    # Credits to Sayak Paul for this beautiful expression
    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(params_to_train, lr=cfg.learning_rate)

    train_model(model, optimizer, cfg, train_dataloader)

    # Push the checkpoint to hub
    lora_model.push_to_hub(cfg.checkpoint_id)
    processor.push_to_hub(cfg.checkpoint_id)

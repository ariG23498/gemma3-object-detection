import logging
import wandb
from functools import partial
import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config import Configuration
from utils import train_collate_function, get_model_with_resize_token_embeddings
import argparse
import albumentations as A
import sys
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_augmentations(cfg):
    # This can be customized
    resize_size = 896
    augmentations = A.Compose([
        A.Resize(height=resize_size, width=resize_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], filter_invalid_bboxes=True))
    return augmentations


def get_dataloaders(processor, cfg):
    logger.info(f"Fetching the dataset: {cfg.dataset_id}")
    
    try:
        logger.info("Attempting to load dataset with trust_remote_code=True")
        dataset = load_dataset(cfg.dataset_id, trust_remote_code=True)
        
        logger.info(f"Available splits: {list(dataset.keys())}")
        
        if "validation" in dataset:
            train_dataset = dataset["train"]
            val_dataset = dataset["validation"]
            logger.info("Found train and validation splits")
        else:
            logger.info("No validation split found. Creating 90/10 split from train data.")
            train_data = dataset["train"]
            train_size = int(0.9 * len(train_data))
            val_size = len(train_data) - train_size
            train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

    except Exception as e:
        logger.error(f"FATAL: Failed to load dataset: {e}")
        logger.error("Please check the dataset ID in config.py and your internet connection.")
        logger.error("If the dataset requires special permissions, ensure you are logged in with `huggingface-cli login`.")
        sys.exit(1)

    train_collate_fn = partial(
        train_collate_function, processor=processor, device=cfg.device, transform=get_augmentations(cfg)
    )
    val_collate_fn = partial(
        train_collate_function, processor=processor, device=cfg.device, transform=None
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, collate_fn=train_collate_fn, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, collate_fn=val_collate_fn, shuffle=False
    )
    return train_dataloader, val_dataloader


@torch.no_grad()
def evaluate_model(model, val_dataloader, device):
    model.eval()
    total_loss = 0
    total_count = 0
    for batch in val_dataloader:
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item() * batch["input_ids"].size(0)
        total_count += batch["input_ids"].size(0)
        
    avg_loss = total_loss / total_count if total_count > 0 else 0
    model.train()
    return avg_loss


def train_model(model, optimizer, cfg, train_dataloader, val_dataloader=None):
    logger.info("Start training")
    global_step = 0
    best_val_loss = float("inf")
    os.makedirs(cfg.best_model_output_dir, exist_ok=True)
    
    for epoch in range(cfg.epochs):
        logger.info(f"Starting epoch {epoch}")
        for idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            if idx % 5 == 0:
                logger.info(f"Epoch: {epoch}, Step: {idx}, Loss: {loss.item():.4f}")
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
            # if idx >= 10:
            #     logger.info("Stopping after 10 steps for testing purposes")
            #     break
                
        if val_dataloader is not None:
            logger.info("Running validation...")
            val_loss = evaluate_model(model, val_dataloader, model.device)
            logger.info(f"Epoch: {epoch} Validation Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(cfg.best_model_output_dir)
                logger.info(f"New best model adapter saved to {cfg.best_model_output_dir}")
    
    return model


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )


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
    parser.add_argument('--attn_imp', type=str, help='attn_implementation to use. eager or flash_attention_2')

    args = parser.parse_args()

    if args.model_id: cfg.model_id = args.model_id
    if args.dataset_id: cfg.dataset_id = args.dataset_id
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.learning_rate: cfg.learning_rate = args.learning_rate
    if args.epochs: cfg.epochs = args.epochs
    if args.checkpoint_id: cfg.checkpoint_id = args.checkpoint_id
    if args.attn_imp: cfg.attn_implementation = args.attn_imp

    logger.info("="*60)
    logger.info("Starting QLoRA Training Test")
    logger.info("="*60)

    logger.info("Step 1: Loading processor...")
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    if args.include_loc_tokens:
        logger.info("Adding location tokens to the tokenizer")
        # This function needs to be defined in utils.py
        # processor = get_processor_with_new_tokens(processor)
        pass
    logger.info("Processor loaded.")

    logger.info("Step 2: Loading and splitting dataset...")
    train_dataloader, val_dataloader = get_dataloaders(processor=processor, cfg=cfg)
    logger.info("Dataset loaded and dataloaders created.")

    logger.info("Step 3: Loading model with QLoRA...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto", 
        attn_implementation=cfg.attn_implementation,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    logger.info("Base model loaded. Preparing for k-bit training...")
    
    model = prepare_model_for_kbit_training(model)
    
    if cfg.lora_target_modules is None:
        import re
        pattern = r'self_attn\.(q_proj|k_proj|v_proj|o_proj)$'
        lora_target_modules = [name for name, _ in model.named_modules() if re.search(pattern, name)]
        if not lora_target_modules:
            logger.warning("No modules found with regex, using default module names")
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        cfg.lora_target_modules = list(set(lora_target_modules))
    
    logger.info(f"Applying LoRA to modules: {cfg.lora_target_modules}")
    peft_config = LoraConfig(
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        r=cfg.lora_r,
        bias="none",
        target_modules=cfg.lora_target_modules,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    if args.include_loc_tokens:
        logger.info("Resizing token embeddings for location tokens...")
        model = get_model_with_resize_token_embeddings(model, processor)
    logger.info("Model ready for training.")

    print_trainable_parameters(model)

    # To enable experiment tracking, uncomment the following lines and run `wandb login`
    # logger.info("Step 4: Setting up optimizer and Weights & Biases...")
    # wandb.init(
    #     project=cfg.project_name,
    #     name=f"{cfg.model_id.replace('/', '_')}-qlora",
    #     config=vars(cfg),
    # )
    
    logger.info("Step 4: Setting up optimizer...")
    model.train()
    params_to_train = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params_to_train, lr=cfg.learning_rate)

    logger.info("Step 5: Starting training loop...")
    train_model(model, optimizer, cfg, train_dataloader, val_dataloader)
    
    # Uncomment the following line to finish logging to W&B
    # wandb.finish()
    
    logger.info("="*60)
    logger.info("✅ QLoRA TRAINING TEST COMPLETED SUCCESSFULLY!")
    logger.info("✅ Model trained with 4-bit quantization!")
    logger.info("="*60)
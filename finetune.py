import logging
import wandb
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import LoraConfig, get_peft_model

from config import Configuration
from utils import train_collate_function
import argparse
import albumentations as A
import yaml
from tqdm import tqdm
from transformers import BitsAndBytesConfig  
from peft import prepare_model_for_kbit_training  







logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

augmentations = A.Compose([
    A.Resize(height=896, width=896),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], filter_invalid_bboxes=True))

def get_dataloader(processor, args, dtype):
    logger.info("Fetching the dataset")
    train_dataset = load_dataset(cfg.dataset_id, split="train")
    train_collate_fn = partial(
        train_collate_function, processor=processor, dtype=dtype, transform=augmentations
    )

    logger.info("Building data loader")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
    )
    return train_dataloader

def train_model(model, optimizer, cfg, train_dataloader):
    logger.info("Start training")
    global_step = 0
    
    
    epoch_pbar = tqdm(range(cfg.epochs) , desc = "Epochs" , position= 0)
    
    
    for epoch in epoch_pbar:
        
        epoch_pbar.set_description(f"Epoch {epoch+1}/{cfg.epochs}")
        
        
        
        batch_pbar = tqdm(train_dataloader, desc="Batches", leave=False, position=1)
        
        for idx, batch in enumerate(batch_pbar):
            outputs = model(**batch.to(model.device))
            loss = outputs.loss
            if idx % 100 == 0:
                logger.info(f"Epoch: {epoch} Iter: {idx} Loss: {loss.item():.4f}")
                wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=global_step)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        batch_pbar.close()
        
    epoch_pbar.close()
    return model


def get_peft_config(peft_type: str, config_dict: dict) -> LoraConfig:
    """Factory method to create PEFT config based on type"""
    common_config = {
        "r": config_dict["r"],
        "lora_alpha": config_dict["lora_alpha"],
        "target_modules": config_dict["target_modules"],
        "lora_dropout": config_dict["lora_dropout"],
        "bias": config_dict["bias"],
        "task_type": config_dict["task_type"],
    }
    
    if peft_type == "qlora":
        # Add QLoRA specific configurations if needed
        common_config.update({
            "use_dora": config_dict.get("use_dora", False),  # DORA: Weight-Decomposed Low-Rank Adaptation
        })
    return LoraConfig(**common_config)
    
    

if __name__ == "__main__":
    cfg = Configuration()

    # Get values dynamically from user
    parser = argparse.ArgumentParser(description="Training for PaLiGemma")
    parser.add_argument('--model_id', type=str, required=True, default=cfg.model_id, help='Enter Huggingface Model ID')
    parser.add_argument('--dataset_id', type=str, required=True ,default=cfg.dataset_id, help='Enter Huggingface Dataset ID')
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size, help='Enter Batch Size')
    parser.add_argument('--lr', type=float, default=cfg.learning_rate, help='Enter Learning Rate')
    parser.add_argument('--checkpoint_id', type=str, required=True, default=cfg.checkpoint_id, help='Enter Huggingface Repo ID to push model')


    parser.add_argument('--peft_type', type=str, required=True, choices=["lora" , "qlora"] ,help='Enter peft type .for eg. lora , qlora ..etc')
    parser.add_argument('--peft_config', type=str, default="peft_configs/lora_configs.yaml",
                      help="Path to peft config YAML file")





    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model_id)
    train_dataloader = get_dataloader(processor=processor, args=args, dtype=cfg.dtype)

    logger.info("Getting model")
    
    if args.peft_type == 'qlora':
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=cfg.dtype,
            )
    else:
        bnb_config = None
    
    
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.dtype,
        device_map="cpu",
        attn_implementation="eager",
        quantization_config=bnb_config if args.peft_type == "qlora" else None
    )
    logger.info(f"Loading PeFT config from {args.peft_type}")




    if args.peft_type == "lora":
        with open(args.peft_config) as f:
            lora_config_dict = yaml.safe_load(f)[f"{args.peft_type}_config"]
        
        lora_config = get_peft_config(peft_type=args.peft_type , config_dict=lora_config_dict)
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()


    if args.peft_type =='qlora':
        
        
        with open(args.peft_config) as f:
            qlora_config_dict = yaml.safe_load(f)[f"{args.peft_type}_config"]
        
        peft_config = get_peft_config(args.peft_type, qlora_config_dict)  
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
            

    model.train()
    model.to(cfg.device)

    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(params_to_train, lr=args.lr)

    wandb.init(
        project=cfg.project_name,
        name=cfg.run_name if hasattr(cfg, "run_name") else f"{args.peft_type} run",
        config=vars(cfg),
    )

    train_model(model, optimizer, cfg, train_dataloader)

    # Push the checkpoint to hub
    model.push_to_hub(cfg.checkpoint_id)
    processor.push_to_hub(cfg.checkpoint_id)

    wandb.finish()
    logger.info("Train finished")
import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw
import torch
from utils.create_dataset import format_objects
from transformers import AutoModel, AutoTokenizer  # Change to your model class if needed
from peft import PeftModel, PeftConfig

def parse_paligemma_label(label, width, height):
    # Extract location codes
    loc_pattern = r"<loc(\d{4})>"
    locations = [int(loc) for loc in re.findall(loc_pattern, label)]

    # Extract category (everything after the last location code)
    category = label.split(">")[-1].strip()

    # Convert normalized locations back to original image coordinates
    # Order in PaliGemma format is: y1, x1, y2, x2
    y1_norm, x1_norm, y2_norm, x2_norm = locations

    # Convert normalized coordinates to actual coordinates
    x1 = (x1_norm / 1024) * width
    y1 = (y1_norm / 1024) * height
    x2 = (x2_norm / 1024) * width
    y2 = (y2_norm / 1024) * height

    return category, [x1, y1, x2, y2]


def visualize_bounding_boxes(image, label, width, height, name):
    # Create a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    # Parse the label
    category, bbox = parse_paligemma_label(label, width, height)

    # Draw the bounding box
    draw.rectangle(bbox, outline="red", width=2)

    # Add category label
    draw.text((bbox[0], max(0, bbox[1] - 10)), category, fill="red")

    # Show the image
    plt.figure(figsize=(10, 6))
    plt.imshow(draw_image)
    plt.axis("off")
    plt.title(f"Bounding Box: {category}")
    plt.tight_layout()
    plt.savefig(name)
    plt.show()
    plt.close()

def train_collate_function_unsloth(batch_of_samples, tokenizer, dtype, transform=None):
    """
    unsloth
    """
    images = []
    prompts = []
    for sample in batch_of_samples:
        if transform:
            transformed = transform(
                image=np.array(sample["image"]),
                bboxes=sample["objects"]["bbox"],
                category_ids=sample["objects"]["category"]
            )
            sample["image"] = transformed["image"]
            sample["objects"]["bbox"] = transformed["bboxes"]
            sample["objects"]["category"] = transformed["category_ids"]
            sample["height"] = sample["image"].shape[0]
            sample["width"] = sample["image"].shape[1]
            sample['label_for_paligemma'] = format_objects(sample)['label_for_paligemma'] 
        images.append([sample["image"]])
        prompts.append(
            f"{tokenizer.boi_token} detect \n\n{sample['label_for_paligemma']} {tokenizer.eos_token}"
        )

    # Use tokenizer directly (Unsloth tokenizer supports vision inputs for Gemma3)
    batch = tokenizer(
        images=images, 
        text=prompts, 
        return_tensors="pt", 
        padding=True
    )

    labels = batch["input_ids"].clone()

    # Mask out padding, image tokens, and other special tokens from loss
    image_token_id = [
        tokenizer.tokenizer.convert_tokens_to_ids(tokenizer.boi_token)
    ]
    labels[labels == tokenizer.pad_token_id] = -100
    for tok_id in image_token_id:
        labels[labels == tok_id] = -100
    labels[labels == 262144] = -100  # If this ID is used for your "unused" special token

    batch["labels"] = labels
    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(dtype)

    return batch

def train_collate_function(batch_of_samples, processor, dtype, transform=None):
    images = []
    prompts = []
    for sample in batch_of_samples:
        if transform:
            transformed = transform(image=np.array(sample["image"]), bboxes=sample["objects"]["bbox"], category_ids=sample["objects"]["category"])
            sample["image"] = transformed["image"]
            sample["objects"]["bbox"] = transformed["bboxes"]
            sample["objects"]["category"] = transformed["category_ids"]
            sample["height"] = sample["image"].shape[0]
            sample["width"] = sample["image"].shape[1]
            sample['label_for_paligemma'] = format_objects(sample)['label_for_paligemma'] 
        images.append([sample["image"]])
        prompts.append(
            f"{processor.tokenizer.boi_token} detect \n\n{sample['label_for_paligemma']} {processor.tokenizer.eos_token}"
        )

    batch = processor(images=images, text=prompts, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels

    # List from https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora
    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels

    batch["pixel_values"] = batch["pixel_values"].to(
        dtype
    )  # to check with the implementation
    return batch


def test_collate_function(batch_of_samples, processor, dtype):
    images = []
    prompts = []
    for sample in batch_of_samples:
        images.append([sample["image"]])
        prompts.append(f"{processor.tokenizer.boi_token} detect \n\n")

    batch = processor(images=images, text=prompts, return_tensors="pt", padding=True)
    batch["pixel_values"] = batch["pixel_values"].to(
        dtype
    )  # to check with the implementation
    return batch, images

def str2bool(v):
    """
    Helper function to parse boolean values from cli arguments
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def push_to_hub(model, cfg, tokenizer=None, is_lora=False):
    """
    Push model to huggingface
    """
    push_kwargs = {}
    if tokenizer is not None:
        push_kwargs['tokenizer'] = tokenizer
    model.push_to_hub(cfg.checkpoint_id, **push_kwargs)
    if tokenizer is not None:
        tokenizer.push_to_hub(cfg.checkpoint_id)

def save_best_model(model, cfg, tokenizer=None, is_lora=False, logger=None):
    """Save LoRA adapter or full model based on config."""
    save_path = f"checkpoints/{cfg.checkpoint_id}_best"
    os.makedirs(save_path, exist_ok=True)
    if is_lora:
        if logger: logger.info(f"Saving LoRA adapter to {save_path}")
        model.save_pretrained(save_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)
    else:
        if logger: logger.info(f"Saving full model weights to {save_path}.pt")
        torch.save(model.state_dict(), f"{save_path}.pt")


def load_saved_model(cfg, is_lora=False, device=None, logger=None):
    """
    Load LoRA adapter or full model based on config.
    Returns (model, tokenizer)
    """
    save_path = f"checkpoints/{cfg.checkpoint_id}_best"
    tokenizer = None

    if is_lora:
        if logger: logger.info(f"Loading LoRA adapter from {save_path}")
        # Load base model first, then LoRA weights
        base_model = AutoModel.from_pretrained(cfg.model_id, device_map=device or "auto")
        model = PeftModel.from_pretrained(base_model, save_path, device_map=device or "auto")
        if os.path.exists(os.path.join(save_path, "tokenizer_config.json")):
            tokenizer = AutoTokenizer.from_pretrained(save_path)
    else:
        if logger: logger.info(f"Loading full model weights from {save_path}.pt")
        model = AutoModel.from_pretrained(cfg.model_id, device_map=device or "auto")
        model.load_state_dict(torch.load(f"{save_path}.pt", map_location=device or "cpu"))
        if os.path.exists(os.path.join(save_path, "tokenizer_config.json")):
            tokenizer = AutoTokenizer.from_pretrained(save_path)
    return model, tokenizer
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw

from create_dataset import format_objects

from transformers import AutoTokenizer, AutoProcessor
from config import Configuration
cfg = Configuration()

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


def get_tokenizer_with_new_tokens():
    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    # Get original sizes
    original_vocab_size = tokenizer.vocab_size
    original_total_size = len(tokenizer)

    print(f"Original vocab size (pretrained): {original_vocab_size}")
    print(f"Original total tokenizer size (includes added tokens): {original_total_size}")

    # Add new location tokens
    location_tokens = [f"<loc{i:04}>" for i in range(1024)]
    added_tokens_count = tokenizer.add_tokens(location_tokens, special_tokens=True)

    # Get updated sizes
    new_total_size = len(tokenizer)

    print(f"Number of new tokens added: {added_tokens_count}")
    print(f"New total tokenizer size: {new_total_size}")

    # Attach updated tokenizer to processor if needed
    processor.tokenizer = tokenizer

    # Update the model's embedding size
    # model.resize_token_embeddings(len(tokenizer))
    return processor, tokenizer

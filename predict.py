import os
from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BlipForConditionalGeneration, Gemma3ForConditionalGeneration

from config import Configuration
from utils import test_collate_function, visualize_bounding_boxes
import argparse

os.makedirs("outputs", exist_ok=True)

model_class_map = [
    (lambda name: "gemma" in name, Gemma3ForConditionalGeneration),
    (lambda name: "blip" in name, BlipForConditionalGeneration),
    (lambda name: "kimi" in name, AutoModelForCausalLM),
]

def parse_args():
    parser = argparse.ArgumentParser(description="Fine Tune Gemma3 for Object Detection")
    parser.add_argument("--model", type=str, help="Model checkpoint identifier")
    return parser.parse_args()

def get_model_class(model_name):
    model_name = model_name.lower()
    for condition, model_class in model_class_map:
        if condition(model_name):
            return model_class
    return AutoModelForSeq2SeqLM

def get_dataloader(processor):
    test_dataset = load_dataset(cfg.dataset_id, split="test")
    test_collate_fn = partial(
        test_collate_function, processor=processor, dtype=cfg.dtype
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, collate_fn=test_collate_fn
    )
    return test_dataloader

if __name__ == "__main__":
    args = parse_args()
    cfg = Configuration()
    if args.model:
        cfg.model_id = args.model
        
    processor = AutoProcessor.from_pretrained(cfg.checkpoint_id)
    model_class = get_model_class(cfg.model_id)
    model = model_class.from_pretrained(
        cfg.checkpoint_id,
        torch_dtype=cfg.dtype,
        device_map="cpu",
        )
 
    model.eval()
    model.to(cfg.device)

    test_dataloader = get_dataloader(processor=processor)
    sample, sample_images = next(iter(test_dataloader))
    sample = sample.to(cfg.device)

    generation = model.generate(**sample, max_new_tokens=100)
    decoded = processor.batch_decode(generation, skip_special_tokens=True)

    file_count = 0
    for output_text, sample_image in zip(decoded, sample_images):
        image = sample_image[0]
        width, height = image.size
        visualize_bounding_boxes(
            image, output_text, width, height, f"outputs/output_{file_count}.png"
        )
        file_count += 1

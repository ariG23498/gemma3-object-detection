import os
from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from config import Configuration
from utils import test_collate_function, visualize_bounding_boxes
import albumentations as A

os.makedirs("outputs", exist_ok=True)

def get_augmentations(cfg):
    if "SmolVLM" in cfg.model_id:
        resize_size = 512
    else:
        resize_size = 896

    augmentations = A.Compose([
        A.Resize(height=resize_size, width=resize_size)
    ])
    return augmentations

def get_dataloader(processor, cfg):
    test_dataset = load_dataset(cfg.dataset_id, split="test")
    test_collate_fn = partial(
        test_collate_function, processor=processor, device=cfg.device, transform=get_augmentations(cfg)
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, collate_fn=test_collate_fn
    )
    return test_dataloader


if __name__ == "__main__":
    cfg = Configuration()
    processor = AutoProcessor.from_pretrained(cfg.checkpoint_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.checkpoint_id,
        torch_dtype=cfg.dtype,
        device_map="cpu",
    )
    model.eval()
    model.to(cfg.device)

    test_dataloader = get_dataloader(processor=processor, cfg=cfg)
    sample, sample_images = next(iter(test_dataloader))
    sample = sample.to(cfg.device)

    generation = model.generate(**sample, max_new_tokens=100)
    decoded = processor.batch_decode(generation, skip_special_tokens=True)

    file_count = 0
    for output_text, sample_image in zip(decoded, sample_images):
        image = sample_image[0]
        print(image)
        print(type(image))
        width, height = image.size
        visualize_bounding_boxes(
            image, output_text, width, height, f"outputs/output_{file_count}.png"
        )
        file_count += 1

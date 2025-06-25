import os
from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from config import Configuration
from utils import test_collate_function, visualize_bounding_boxes, get_last_checkpoint_step

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import albumentations as A

os.makedirs("outputs", exist_ok=True)

augmentations = A.Compose([
    A.Resize(height=896, width=896),
    ],
    bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['category_ids'],
        filter_invalid_bboxes=True,
        clip=True,
    )
)


def get_dataloader(processor):
    test_dataset = load_dataset(cfg.dataset_id, split="val")
    test_collate_fn = partial(
        test_collate_function, processor=processor, dtype=cfg.dtype, transform=augmentations,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, collate_fn=test_collate_fn
    )
    return test_dataloader


if __name__ == "__main__":
    cfg = Configuration()

    accelerator = Accelerator(
        project_config=ProjectConfiguration(
            project_dir=f"{cfg.project_dir}/{cfg.run_name}",
            logging_dir=f"{cfg.project_dir}/{cfg.run_name}/{cfg.log_dir}",
            automatic_checkpoint_naming = cfg.automatic_checkpoint_naming,
        ),
    )

    processor = AutoProcessor.from_pretrained(cfg.checkpoint_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.dtype,
        device_map="cpu",
    )

    test_dataloader = get_dataloader(processor=processor)

    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )
    model.eval()

    check_point_number = get_last_checkpoint_step(accelerator)
    global_step = check_point_number * cfg.checkpoint_interval +1
    accelerator.project_configuration.iteration = check_point_number + 1
    accelerator.load_state()

    sample, sample_images = next(iter(test_dataloader))
    generation = model.generate(**sample, max_new_tokens=1000)
    decoded = processor.batch_decode(generation, skip_special_tokens=True)
    file_count = 0
    for output_text, sample_image in zip(decoded, sample_images):
        image = sample_image[0]
        height, width, _ = image.shape
        try:
            visualize_bounding_boxes(
                image, output_text, width, height, f"outputs/output_{file_count}.png"
            )
        except:
            print("failed to generate correct detection format.")
        file_count += 1
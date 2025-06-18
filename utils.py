import re
import os
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from PIL import ImageDraw, Image, ImageFont
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=20)


from create_dataset import format_objects

def parse_paligemma_labels(labels, width, height):
    # assuming <locx><locx><locx><locx> cat0; <locx><locx><locx><locx> cat1 ...
    categories, cords = [],[]
    for label in labels.split(";"):
        category, cord = parse_paligemma_label(label, width, height)
        categories.append(category)
        cords.append(cord)
    return categories, cords

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


def visualize_bounding_boxes(image, labels, width, height, name):
    # Create a copy of the image to draw on
    draw_image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(draw_image)

    # Parse the label
    cats, bboxs = parse_paligemma_labels(labels, width, height)

    for cat, bbox in zip(cats, bboxs):
        # Draw the bounding box
        draw.rectangle(bbox, outline="red", width=2)

        # Add category label
        draw.text((bbox[0], max(0, bbox[1] - 20)), cat, fill="red", font=font)

    # Show the image
    plt.figure(figsize=(10, 6))
    plt.imshow(draw_image)
    plt.axis("off")
    plt.title(f"Dets & Cats")
    plt.tight_layout()
    plt.savefig(name)
    plt.show()
    plt.close()


def train_collate_function(batch_of_samples, processor, dtype, transform=None, return_images=False):
    # @sajjad: need to set a max number of detections to avoid GPU OOM
    MAX_DETS = 50
    images = []
    prompts = []

    for sample in batch_of_samples:
        if transform:
            transformed = transform(
                image=np.array(sample["image"]),
                bboxes=sample["objects"]["bbox"],
                category_ids=sample["objects"]["category"],
            )

            # 1. Fix image shape to 3 channels
            img = transformed["image"]
            if img.ndim == 2:
                img = img[:, :, np.newaxis].repeat(3, axis=2)
            sample["image"] = img

            # 2. Grab the full lists of bboxes & category_ids
            bboxes = transformed["bboxes"]
            cats   = transformed["category_ids"]

            # 3. Randomly sample up to MAX_DETS
            num_objs = len(bboxes)
            # Always sample `sample_size = min(num_objs, MAX_DETS)`:
            sample_size = min(num_objs, MAX_DETS)
            # Since sample_size <= num_objs, we can safely sample without replacement:
            chosen_idx = np.random.choice(num_objs, size=sample_size, replace=False)

            # 4. Subset both lists in exactly the same way
            bboxes = [bboxes[i] for i in chosen_idx]
            cats   = [cats[i]   for i in chosen_idx]

            # 5. Write the (possibly truncated/shuffled) lists back
            sample["objects"]["bbox"]     = bboxes
            sample["objects"]["category"] = cats

            # 6. Update height/width (image may have been transformed)
            sample["height"] = sample["image"].shape[0]
            sample["width"]  = sample["image"].shape[1]

            # 7. Recompute your label string after subsampling
            sample["label_for_paligemma"] = format_objects(sample)["label_for_paligemma"]

        images.append([sample["image"]])
        prompts.append(
            f"{processor.tokenizer.boi_token} detect \n\n"
            f"{sample['label_for_paligemma']} "
            f"{processor.tokenizer.eos_token}"
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
    # @sajjad: what is 262144?
    labels[labels == 262144] = -100

    batch["labels"] = labels

    batch["pixel_values"] = batch["pixel_values"].to(
        dtype
    )  # to check with the implementation

    if return_images:
        return batch, images
    else:
        return batch


def test_collate_function(batch_of_samples, processor, dtype, transform, return_images=True):
    images = []
    prompts = []
    for sample in batch_of_samples:
        transformed = transform(
            image=np.array(sample["image"]),
            bboxes=sample["objects"]["bbox"],
            category_ids=sample["objects"]["category"],
        )
        img = transformed["image"]
        if img.ndim == 2:
            img = img[:, :, np.newaxis].repeat(3, axis=2)
        sample["image"] = img
        images.append([img])
        prompts.append(f"{processor.tokenizer.boi_token} detect \n\n")

    batch = processor(images=images, text=prompts, return_tensors="pt", padding=True)
    batch["pixel_values"] = batch["pixel_values"].to(
        dtype
    )  # to check with the implementation
    if return_images:
        return batch, images
    else:
        return batch

def get_last_checkpoint_step(accelerator):
    input_dir = os.path.join(accelerator.project_dir, "checkpoints")
    folders = [os.path.join(input_dir, folder) for folder in os.listdir(input_dir)]

    def _inner(folder):
        return list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]

    folders.sort(key=_inner)
    input_dir = folders[-1]
    return _inner(input_dir)

def get_augmentations():
    return A.Compose(
        [
            # 0. Affine (shift/scale/rotate) replaces ShiftScaleRotate
            A.Affine(
                translate_percent={"x": 0.0625, "y": 0.0625},  # ±6% shift
                scale=(0.8, 1.2),                               # 80–120% scale
                rotate=(-15, 15),                               # ±15° rotation
                interpolation=1,
                p=0.5
            ),

            # 1. Random “safe” crop that ensures any remaining box still has area
            A.RandomSizedBBoxSafeCrop(
                height=800, width=800,
                erosion_rate=0.0,
                p=0.3
            ),

            # 2. Flips
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),

            # 3. Color‐space augmentations (choose one)
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=15,
                        sat_shift_limit=20,
                        val_shift_limit=15,
                        p=1.0
                    ),
                    A.CLAHE(
                        clip_limit=2.0,
                        tile_grid_size=(8, 8),
                        p=1.0
                    ),
                ],
                p=0.5
            ),

            # 4. Slight RGB shifts
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=0.3
            ),

            # 5. Blur or noise (using GaussNoise, since GaussianNoise isn’t available)
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.GaussNoise(p=1.0),
                ],
                p=0.3
            ),

            # 6. CoarseDropout for “cutout”‐style occlusion
            A.CoarseDropout(
                num_holes_range = (1,8),
                hole_height_range = (32, 64),
                hole_width_range = (32, 64),
                p=0.3
            ),

            # 7. Grid or optical distortions (drop shift_limit in OpticalDistortion)
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    A.OpticalDistortion(distort_limit=0.05, p=1.0),
                ],
                p=0.2
            ),
            # 8. Resize to 896×896
            A.Resize(height=896, width=896),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['category_ids'],
            filter_invalid_bboxes=True,
            clip=True,
        )
    )
import logging
import wandb
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

from config import Configuration
from utils import train_collate_function
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftType


import albumentations as A

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

cfg = Configuration()

augmentations = A.Compose([
    A.Resize(height=896, width=896),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], filter_invalid_bboxes=True))


def initialize_model():
    cfg = Configuration()

    logger.info("Getting model & turning only attention parameters to trainable")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.dtype,
        device_map="cpu",
        attn_implementation="eager",
        quantization_config=bnb_config
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        peft_type=PeftType.LORA,
    )

    qlora_model = get_peft_model(model, lora_config)
    qlora_model.print_trainable_parameters()

    qlora_model.to(cfg.device)

    return qlora_model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available. VRAM measurement requires a CUDA-enabled GPU.")
        return

    qlora_model = initialize_model()

    # Measure VRAM after model is loaded to device
    torch.cuda.synchronize() # Ensure all operations are complete
    initial_vram_allocated_bytes = torch.cuda.memory_allocated(device)
    initial_vram_allocated_mb = initial_vram_allocated_bytes / (1024 ** 2)
    print(f"VRAM allocated after loading model to device: {initial_vram_allocated_mb:.2f} MB")

    # --- Dataset Preparation ---
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    train_dataset = load_dataset(cfg.dataset_id, split="train")
    train_collate_fn = partial(
        train_collate_function, processor=processor, dtype=cfg.dtype, transform=augmentations
    )
   

    batch_sizes_to_test = [int(bs) for bs in cfg.batch_sizes]
    if not batch_sizes_to_test:
        print("Error: No batch sizes provided or parsed correctly.")
        return
    
    num_iterations_for_vram = cfg.num_iterations


    print("\n--- VRAM Measurement ---")
    results = {}

    for bs in batch_sizes_to_test:
        print(f"\nTesting Batch Size: {bs}")

        if len(train_dataset) < bs:
            print(f"Base processed dataset has {len(train_dataset)} samples, "
                  f"not enough for batch size {bs}. Skipping.")
            results[bs] = "Not enough data"
            continue

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=bs,
            collate_fn=train_collate_fn,
            shuffle=True,
        )

        if len(train_dataloader) < num_iterations_for_vram:
            print(f"Dataloader for batch size {bs} yields {len(train_dataloader)} batches, "
                   f"less than requested {num_iterations_for_vram} iterations. Will run available batches.")
            if len(train_dataloader) == 0:
                print(f"Dataloader for batch size {bs} is empty. Skipping.")
                results[bs] = "Dataloader empty"
                continue

        # Reset CUDA memory stats for each batch size test
        torch.cuda.reset_peak_memory_stats(device)        
    
        # Credits to Sayak Paul for this beautiful expression
        qlora_model.train()
        params_to_train = list(filter(lambda x: x.requires_grad, qlora_model.parameters()))
        optimizer = torch.optim.AdamW(params_to_train, lr=cfg.learning_rate)


        try:
            for i, batch in enumerate(train_dataloader):
                if i >= num_iterations_for_vram:
                    break
                
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = qlora_model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            peak_vram_allocated_bytes = torch.cuda.max_memory_allocated(device)
            peak_vram_allocated_mb = peak_vram_allocated_bytes / (1024 ** 2)
            print(f"Peak VRAM allocated for batch size {bs}: {peak_vram_allocated_mb:.2f} MB")
            results[bs] = f"{peak_vram_allocated_mb:.2f} MB"
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                peak_vram_allocated_bytes = torch.cuda.max_memory_allocated(device) # Get max allocated before OOM
                peak_vram_allocated_mb = peak_vram_allocated_bytes / (1024 ** 2)
                print(f"CUDA out of memory for batch size {bs}. ")
                print(f"Peak VRAM allocated before OOM: {peak_vram_allocated_mb:.2f} MB (may be approximate)")
                results[bs] = f"OOM (Peak before OOM: {peak_vram_allocated_mb:.2f} MB)"
            else:
                print(f"An unexpected runtime error occurred for batch size {bs}: {e}")
                results[bs] = f"Error: {e}"
                # raise e # Optionally re-raise for debugging
        
        finally:
            del current_loader, optimizer
            torch.cuda.empty_cache()
        
    print("\n--- Summary of VRAM Usage ---")
    for bs, vram_usage in results.items():
        print(f"Batch Size {bs}: {vram_usage}")


if __name__ == "__main__":
    main()

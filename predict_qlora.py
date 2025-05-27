import os
from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import PeftModel

from config import Configuration
from utils import test_collate_function, visualize_bounding_boxes

os.makedirs("outputs", exist_ok=True)


def get_dataloader(processor):
    test_dataset = load_dataset(cfg.dataset_id, split="test")
    test_collate_fn = partial(
        test_collate_function, processor=processor, dtype=cfg.dtype
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, collate_fn=test_collate_fn
    )
    return test_dataloader


def load_model_for_inference(cfg):
    """Charge le modèle pour l'inférence selon la configuration"""
    
    if cfg.use_qlora:
        # Charger le modèle de base avec quantification
        print("Loading base model with quantization...")
        base_model = Gemma3ForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.dtype,
            device_map="auto",
            quantization_config=cfg.bnb_config,
            trust_remote_code=True,
        )
        
        # Charger les adaptateurs LoRA
        print("Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, cfg.checkpoint_id)
        print("Model loaded with QLoRA adapters")
        
    else:
        # Mode traditionnel : charger le modèle complet
        print("Loading full fine-tuned model...")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            cfg.checkpoint_id,
            torch_dtype=cfg.dtype,
            device_map="auto",
        )
    
    return model


if __name__ == "__main__":
    cfg = Configuration()
    
    # Charger le processeur
    processor = AutoProcessor.from_pretrained(
        cfg.checkpoint_id if not cfg.use_qlora else cfg.model_id
    )
    
    # Charger le modèle selon la configuration
    model = load_model_for_inference(cfg)
    model.eval()

    # Préparer les données de test
    test_dataloader = get_dataloader(processor=processor)
    sample, sample_images = next(iter(test_dataloader))
    
    # Déplacer sur le bon device
    sample = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in sample.items()}

    # Génération
    print("Generating predictions...")
    generation = model.generate(**sample, max_new_tokens=100, do_sample=False)
    decoded = processor.batch_decode(generation, skip_special_tokens=True)

    # Visualisation des résultats
    file_count = 0
    for output_text, sample_image in zip(decoded, sample_images):
        image = sample_image[0]
        width, height = image.size
        
        print(f"Generated text for image {file_count}: {output_text}")
        
        visualize_bounding_boxes(
            image, output_text, width, height, f"outputs/output_{file_count}.png"
        )
        file_count += 1
    
    print(f"Generated {file_count} predictions in outputs/ directory")
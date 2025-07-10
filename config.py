from dataclasses import dataclass
import torch

@dataclass
class Configuration:
    dataset_id: str = "ariG23498/license-detection-paligemma"
    
    project_name: str = "gemma-3-4b-pt-object-detection-qlora-test"
    model_id: str = "google/gemma-3-4b-pt"
    checkpoint_id: str = "test/gemma-3-4b-pt-qlora-adapter"
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = "auto"
    attn_implementation = "eager"
    
    # Reduced for testing
    batch_size: int = 1
    learning_rate: float = 2e-05
    epochs = 10  # Set to 1 for a quick test, or a higher value (e.g., 10) for a full training run
    best_model_output_dir: str = "outputs/best_model"
    
    # QLoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = None

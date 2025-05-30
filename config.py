from dataclasses import dataclass

import torch


@dataclass
class Configuration:
    dataset_id: str = "ariG23498/license-detection-paligemma"

    project_name: str = "SmolVLM-256M-Instruct-object-detection-aug" # "gemma-3-4b-pt-object-detection-aug"
    model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct" # "google/gemma-3-4b-pt"
    checkpoint_id: str = "sergiopaniego/SmolVLM-256M-Instruct-object-detection" # "sergiopaniego/gemma-3-4b-pt-object-detection-aug"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = "auto" # Change to torch.bfloat16 for "google/gemma-3-4b-pt"

    batch_size: int = 1 # 8 for "google/gemma-3-4b-pt"
    learning_rate: float = 2e-05
    epochs = 2

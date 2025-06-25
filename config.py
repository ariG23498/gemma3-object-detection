from dataclasses import dataclass

import torch


@dataclass
class Configuration:
    dataset_id: str = "ariG23498/license-detection-paligemma"

    project_name: str = "gemma-3-4b-pt-object-detection-aug" # "SmolVLM-256M-Instruct-object-detection-aug"
    model_id: str = "google/gemma-3-4b-pt" # "HuggingFaceTB/SmolVLM-256M-Instruct"
    checkpoint_id: str = "sergiopaniego/gemma-3-4b-pt-object-detection-loc-tokens" # "sergiopaniego/SmolVLM-256M-Instruct-object-detection"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = "auto" # Change to torch.bfloat16 for "google/gemma-3-4b-pt"

    batch_size: int = 4 # 8 for "google/gemma-3-4b-pt"
    learning_rate: float = 2e-05
    epochs = 2


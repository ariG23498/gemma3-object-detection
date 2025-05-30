from dataclasses import dataclass

import torch


@dataclass
class Configuration:
    dataset_id: str = "savoji/coco-paligemma"

    model_id: str = "google/gemma-3-4b-pt"
    checkpoint_id: str = "savoji/gemma-3-4b-pt-coco"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

    batch_size: int = 1
    learning_rate: float = 2e-05
    epochs = 1

    project_name: str = "gemma3-coco"
    run_name: str  = "coco_00"
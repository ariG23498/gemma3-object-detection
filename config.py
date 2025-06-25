from dataclasses import dataclass

import torch


@dataclass
class Configuration:
    dataset_id: str = "detection-datasets/coco"

    model_id: str = "google/gemma-3-4b-pt"
    checkpoint_id: str = "savoji/gemma-3-4b-pt-coco"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

    batch_size: int = 32
    learning_rate: float = 2e-05
    epochs = 10

    project_name: str = "gemma3-coco"
    run_name: str  = "coco_aug"
    project_dir: str = "runs"
    log_dir: str  = "logs"

    checkpoint_interval: int = 50000
    log_interval: int = 100
    automatic_checkpoint_naming: bool = True
    resume: bool = True
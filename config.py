from dataclasses import dataclass

import torch


@dataclass
class Configuration:
    dataset_id: str = "ariG23498/coco-detection-strings"
    "ariG23498/coco-detection-strings"
    "ariG23498/license-detection-paligemma"

    model_id: str = "google/gemma-3-4b-pt"
    checkpoint_id: str = "savoji/gemma-3-4b-pt-coco-detection"
    "ariG23498/gemma-3-4b-pt-object-detection"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

    batch_size: int = 1
    learning_rate: float = 2e-05
    epochs = 1

    project_name: str = "gemma3-coco-detection"
    run_name: str  = "coco_00"
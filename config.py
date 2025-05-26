from dataclasses import dataclass

import torch


@dataclass
class Configuration:
    dataset_id: str = "ariG23498/license-detection-paligemma"

    model_id: str = "google/gemma-3-4b-pt"
    checkpoint_id: str = "sergiopaniego/gemma-3-4b-pt-object-detection-aug"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

    batch_size: int = 8
    learning_rate: float = 2e-05
    epochs = 2
    
    @staticmethod
        def from_args():
            parser = argparse.ArgumentParser()
            parser.add_argument('--dataset', type=str, help='Hugging Face dataset ID')
            parser.add_argument('--config', type=str, default=None, help='Dataset configuration name')
            args, _ = parser.parse_known_args()
            cfg = Configuration()
            if args.dataset:
                cfg.dataset_id = args.dataset
            if args.config:
                cfg.dataset_config = args.config
            return cfg

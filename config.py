from dataclasses import dataclass
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig


@dataclass
class Configuration:
    # Identifiants
    dataset_id: str = "ariG23498/license-detection-paligemma"
    model_id: str = "google/gemma-3-4b-pt"
    checkpoint_id: str = "sergiopaniego/gemma-3-4b-pt-object-detection-qlora"

    # Infos projet (ajouté pour wandb)
    project_name: str = "gemma3-detection"
    run_name: str = "run-qlora"

    # Entraînement
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 2

    # Activation QLoRA
    use_qlora: bool = True

    # Paramètres LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    @property
    def bnb_config(self):
        """Configuration de quantification 4-bit pour QLoRA"""
        if not self.use_qlora:
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=True,
        )

    @property
    def lora_config(self):
        """Configuration LoRA utilisée dans le setup du modèle"""
        if not self.use_qlora:
            return None
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=None  # sera défini dans train.py selon le modèle
        )

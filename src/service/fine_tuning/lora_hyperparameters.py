import math
from typing import Dict


class LoRAHyperparameters:
    """Utility class for calculating optimal LoRA hyperparameters."""

    @staticmethod
    def calculate(
            model_size: int,
            dataset_size: int,
            k_r: float = 2.0,
            k_alpha: float = 1.0,
            min_r: int = 4,
            max_r: int = 64
    ) -> Dict[str, int]:
        """
        Calculate optimal LoRA hyperparameters.

        Args:
            model_size (int): Total number of parameters in the model
            dataset_size (int): Number of samples in the dataset
            k_r (float): Scaling constant for rank
            k_alpha (float): Scaling constant for lora_alpha
            min_r (int): Minimum allowed rank
            max_r (int): Maximum allowed rank

        Returns:
            Dict[str, int]: Dictionary with optimal 'r' and 'lora_alpha'

        Raises:
            ValueError: For invalid input parameters
        """
        # Input validation
        if model_size <= 0 or dataset_size <= 0:
            raise ValueError("Model size and dataset size must be positive")

        if k_r <= 0 or k_alpha <= 0:
            raise ValueError("Scaling constants must be positive")

        # Calculate rank
        param_dataset_ratio = model_size / dataset_size
        r = int(k_r * (param_dataset_ratio ** 0.5) * (1 + math.log(param_dataset_ratio + 1)))
        r = max(min(r, max_r), min_r)

        # Calculate scaling factor
        r = max(r, 1)  # Fallback to at least 1

        # Base alpha computation
        base_alpha = k_alpha * (model_size / (r * dataset_size))

        # Rank sensitivity adjustment
        rank_sensitivity = math.sqrt(max_r / r)
        model_size_factor = math.log(model_size + 1) / math.log(1e9 + 1)

        # Final alpha computation
        lora_alpha = int(base_alpha * rank_sensitivity * model_size_factor)
        lora_alpha = max(lora_alpha, 1)

        return {'r': r, 'lora_alpha': lora_alpha}


lora_hyperparameters = LoRAHyperparameters()

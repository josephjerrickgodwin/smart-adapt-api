import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from trl import SFTTrainer
from datasets import Dataset
from typing import List, Optional, Generator
import os

from src.service.storage_manager import storage_manager


class ModelFineTuner:
    """Utility class for fine-tuning language models."""

    _adapter_dir = 'adapters'
    _logs_dir = 'logs'

    @classmethod
    def prepare_model(
            cls,
            model_name: str,
            r: int,
            lora_alpha: int
    ):
        """
        Prepare the base model for fine-tuning.

        Args:
            model_name (str): Name of the model to load
            r (int): LoRA rank
            lora_alpha (int): LoRA scaling factor

        Returns:
            Tuple of prepared model and tokenizer
        """
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )

        # LoRA configuration
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['q_proj', 'k_proj']
        )

        # Apply LoRA to model
        base_model = get_peft_model(base_model, peft_config)

        return base_model, tokenizer

    @classmethod
    async def fine_tune(
            cls,
            dataset: List[str],
            model_name: str,
            username: str,
            r: int,
            lora_alpha: int,
            num_epochs: int = 10,
            learning_rate: float = 2e-4
    ) -> Generator[str, None, None]:
        """
        Fine-tune the model with the given dataset.

        Args:
            dataset (List[str]): Preprocessed text chunks
            model_name (str): Name of the model to fine-tune
            username (str): Name of the user
            r (int): LoRA rank
            lora_alpha (int): LoRA scaling factor
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate for training

        Yields:
            str: Progress updates during fine-tuning
        """
        # Prepare the data storage
        data_path = await storage_manager.get_user_data_path(username)

        # Define the adapter and logs path
        adapter_path = os.path.join(data_path, cls._adapter_dir)
        logs_dir_path = os.path.join(data_path, cls._logs_dir)

        # Prepare dataset
        hf_dataset = Dataset.from_list([{'text': text} for text in dataset])

        # Prepare model and tokenizer
        yield "Preparing model and tokenizer..."
        base_model, tokenizer = cls.prepare_model(model_name, r, lora_alpha)

        # Training arguments
        training_arguments = TrainingArguments(
            output_dir=logs_dir_path,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            optim="paged_adamw_32bit",
            num_train_epochs=num_epochs,
            eval_strategy="steps",
            eval_steps=0.2,
            logging_steps=1,
            warmup_steps=10,
            logging_strategy="steps",
            learning_rate=learning_rate,
            fp16=False,
            bf16=False,
            group_by_length=True,
            report_to="none"
        )

        # Initialize trainer
        yield "Initializing trainer..."
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=hf_dataset,
            eval_dataset=hf_dataset,
            peft_config=None,  # Already applied
            max_seq_length=256,
            dataset_text_field="text",
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False,
        )

        # Perform training
        yield "Starting fine-tuning process..."
        trainer.train()

        # Save model
        yield "Saving fine-tuned model..."

        os.makedirs(adapter_path, exist_ok=True)
        trainer.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        yield "Fine-tuning complete!"

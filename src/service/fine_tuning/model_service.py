import logging
import os

from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from src.service.storage_manager import storage_manager

load_dotenv()

logger = logging.getLogger(__name__)

# Load the Sentence Transformer Model
LLM_ID = str(os.getenv('LLM_MODEL_ID'))


class ModelService:
    """Main service for model fine-tuning and inference."""

    def __init__(self):
        self.model_id = LLM_ID
        self.data_model_name = 'adapters'

    async def get_inference_pipeline(self, email: str, use_lora: bool = False):
        """
        Get the inference pipeline for the model.

        Args:
            email (str): Email address of the user
            use_lora (bool): Whether to use lora for inference

        Returns:
            text-generation pipeline
        """
        # Check for existing LoRA adapters
        lora_exist = await storage_manager.exists(
            email=email,
            file_name=self.data_model_name
        )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto"
        )

        # Load the model with LoRA if the weights are available
        if lora_exist and use_lora:
            user_data_weights = await storage_manager.read(
                email=email,
                file_name=self.data_model_name
            )
            peft_config = PeftConfig.from_pretrained(user_data_weights)
            model = PeftModel.from_pretrained(model, peft_config)

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )


model_service = ModelService()

import gc
import logging
import os
from threading import Thread
from typing import List, Generator

import torch
from dotenv import load_dotenv

from datasets import Dataset
from peft import (
    PeftModel,
    PeftConfig,
    LoraConfig,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from src.exception.inference_disabled_error import InferenceDisabledError
from src.service import prompt_service
from src.service.TextStreamer import SmartAdaptTextStreamer
from src.service.storage_manager import storage_manager
from src.service.streaming_service import streaming_service

load_dotenv()

logger = logging.getLogger(__name__)

# Load the Sentence Transformer Model and the HF token
REASONING_MODEL_ID = str(os.getenv('REASONING_MODEL_ID'))
HF_TOKEN = str(os.getenv('HF_TOKEN'))


class ModelService:
    """
    Main service for LLM inference.
    """
    def __init__(self):
        self.token = HF_TOKEN
        self.reasoning_model_id = REASONING_MODEL_ID
        self.data_model_name = 'adapters'
        self.logs_dir = 'logs'

        # Define model and tokenizer for reasoning
        self.tokenizer = None
        self.model = None

        # Define model and tokenizer for text completion
        self.infer_tokenizer = None
        self.infer_model = None

        # Define the device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Define the current user
        self.current_user = None
        self.lora_loaded = False
        self.inference_enabled = True

        # Load weights
        self._load_weights()

    def flush(self):
        # Disable inference temporarily
        self.inference_enabled = False

        # Unload the thinker and inference models along with the tokenizers
        self.tokenizer = None
        self.model = None
        self.infer_tokenizer = None
        self.infer_model = None

        # Reset adapter and user details
        self.current_user = None
        self.lora_loaded = False

        # Clear cache
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        # Load the model again
        self._load_weights()

        # Reset the inference status
        self.inference_enabled = True

    async def _load_adapter(self, email: str, merge_and_unload: bool = False):
        """
        Get the inference pipeline for the model.

        Args:
            email (str): Email address of the user

        Returns:
            text-generation pipeline
        """
        # Check for the current user
        if not self.current_user:
            self.current_user = email

        # If the user adapter is already exist, skip loading it again
        if self.lora_loaded:
            if email == self.current_user:
                logger.info('LoRA adapter already loaded. Skipping auto assign')
                return
            else:
                logger.info('Removing LoRA adapter of the previous user')
                self.model.delete_adapter(self.current_user)
                self.lora_loaded = False

        # Check for LoRA adapters for the current user in the local storage
        lora_exist = await storage_manager.exists(
            email=email,
            file_name=self.data_model_name
        )
        logger.info(f"{'Found LoRA adapters and loading to the model' if lora_exist else 'Found no LoRA adapters'}")

        # Load the model with LoRA, if the weights are available
        if lora_exist:
            user_data_weights = await storage_manager.read(
                email=email,
                file_name=self.data_model_name
            )
            peft_config = PeftConfig.from_pretrained(user_data_weights)
            self.model = PeftModel.from_pretrained(
                self.model,
                peft_config,
                adapter_name=email
            ).to(self.device)

            # Merge the adapter permanently
            if merge_and_unload:
                self.model = self.model.merge_and_unload()

            self.lora_loaded = True
            logger.info('Successfully loaded LoRA adapters')

        # Assign the current user
        self.current_user = email

    async def start_inference(
            self,
            user_id: str,
            user_query: str,
            history: list,
            context: str = '',
            index: int = 0,
            message_id: int = 0,
            parent_id: int = 0,
            max_new_tokens: int = 2048
    ):
        if not self.inference_enabled:
            raise InferenceDisabledError("The model is being trained. Please try again later!")

        # Load user adapter
        await self._load_adapter(user_id)

        # Format messages
        system_prompt_for_thinker_model = prompt_service.system_prompt_for_thinker_model

        # Add context to reasoning
        if context:
            context = f'===================\nKnown Facts for thinking\n{context}\n==================='
            updated_messages = f'{system_prompt_for_thinker_model}\n{context}\nQuery: {user_query}\nThought:'
        else:
            updated_messages = f'{system_prompt_for_thinker_model}\nQuery: {user_query}\nThought:'

        # Tokenize the messages
        tokens = self.tokenizer(
            updated_messages,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.device)

        # Define the streamer
        streamer = SmartAdaptTextStreamer(
            tokenizer=self.tokenizer,
            text_type="thinking",
            model='',
            index=index,
            message_id=message_id,
            parent_id=parent_id,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Define the generation kwargs
        thinker_kwargs = dict(
            tokens.input_ids,
            max_new_tokens=max_new_tokens,
            attention_mask=tokens.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            streamer=streamer
        )

        # Define the thread
        thread = Thread(target=self.model.generate, kwargs=thinker_kwargs)

        # Start the thread
        thread.start()

        # Start streaming the reasoning
        async for new_text in streamer:
            await streaming_service.stream(new_text)

        # Get the streamed message from the `thinker` model
        reasoning = streamer.get_response()

        # Update the streamer for `text` streaming
        streamer.update_text_type('text')

        # Re-format the user message with the context
        custom_user_message = [{
            "role": "user",
            "content": f'{context}\n\nExpert Message:\n\n{reasoning}\n\nQuestion: {user_query}'.strip()
        }]

        # Define the system prompt for inference
        system_prompt_for_inference = prompt_service.inference_system_prompt
        system_message = [{
            "role": "system",
            "content": system_prompt_for_inference
        }]
        messages = system_message + history + custom_user_message

        # Apply chat template
        updated_messages = self.infer_tokenizer.apply_chat_template(messages, tokenize=False)

        # Tokenize the messages
        inference_tokens = self.infer_tokenizer(
            updated_messages,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.device)

        # Define the generation kwargs
        generation_kwargs = dict(
            inference_tokens.input_ids,
            max_new_tokens=max_new_tokens,
            attention_mask=inference_tokens.attention_mask,
            pad_token_id=self.infer_tokenizer.eos_token_id,
            do_sample=True,
            streamer=streamer
        )

        # Define the thread
        thread = Thread(target=self.infer_model.generate, kwargs=generation_kwargs)

        # Start the thread
        thread.start()

        # Start streaming the reasoning
        async for new_text in streamer:
            await streaming_service.stream(new_text)

    def _load_weights(self):
        """
        Loads the `thinker` and `inference` model instances
        """
        # Load the tokenizer
        logger.info('Started loading the `thinker` tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.reasoning_model_id,
            token=self.token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model
        logger.info('Started loading the `thinker` model')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.reasoning_model_id,
            token=self.token
        ).to(self.device)

        # Load the inference model and tokenizer
        logger.info('Started loading the `inference` tokenizer')
        self.infer_tokenizer = AutoTokenizer.from_pretrained(
            self.reasoning_model_id,
            token=self.token
        )
        self.infer_tokenizer.pad_token = self.infer_tokenizer.eos_token

        # Load the model
        logger.info('Started loading the `inference` model')
        self.infer_model = AutoModelForCausalLM.from_pretrained(
            self.reasoning_model_id,
            token=self.token
        ).to(self.device)

        logger.info('Successfully loaded the `thinker` and `inference` instances')

    async def prepare_model(
            self,
            user_id: str,
            r: int,
            lora_alpha: int
    ):
        """
            Prepare the base model for fine-tuning.

            Args:
                user_id (str): Unique ID of the user
                r (int): LoRA rank
                lora_alpha (int): LoRA scaling factor

            Returns:
                Tuple of prepared model and tokenizer
        """
        # Load adapters of the current user
        await self._load_adapter(user_id)

        # LoRA configuration
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            use_dora=True,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['q_proj', 'k_proj']
        )

        # Apply LoRA to the model
        self.model = get_peft_model(self.model, peft_config).to(self.device)

    async def fine_tune(
            self,
            user_id: str,
            dataset: List[str],
            r: int,
            lora_alpha: int,
            num_epochs: int = 10,
            max_seq_len: int = 256,
            learning_rate: float = 2e-4,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            report_to: str = 'none'
    ) -> Generator[str, None, None]:
        """
            Fine-tune the model with the given dataset.

            Args:
                user_id (str): Unique ID of the user
                dataset (List[str]): Preprocessed text chunks
                r (int): LoRA rank
                lora_alpha (int): LoRA scaling factor
                num_epochs (int): Number of training epochs
                max_seq_len (int): Maximum sequence length from the dataset
                learning_rate (float): Learning rate for training
                per_device_train_batch_size (int): Training batch size per device
                per_device_eval_batch_size (int): Evaluation batch size per device
                gradient_accumulation_steps (int): Gradient accumulation steps
                report_to (str): Report the training and evaluation results

            Yields:
                str: Progress updates during fine-tuning
        """
        # Disable inference during training
        self.inference_enabled = False

        # Prepare the data storage
        data_path = await storage_manager.get_user_data_path(user_id)

        # Define the adapter and logs path
        adapter_path = os.path.join(data_path, self.data_model_name)
        logs_dir_path = os.path.join(data_path, self.logs_dir)

        # Prepare the dataset
        hf_dataset = Dataset.from_list([{'text': text} for text in dataset])

        # Prepare model and tokenizer
        yield "Preparing model and tokenizer..."
        await self.prepare_model(
            user_id=user_id,
            r=r,
            lora_alpha=lora_alpha
        )

        # Training arguments
        training_arguments = TrainingArguments(
            output_dir=logs_dir_path,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
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
            report_to=report_to
        )

        # Initialize trainer
        yield "Initializing trainer..."
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=hf_dataset,
            eval_dataset=hf_dataset,
            peft_config=None,  # Already applied
            max_seq_length=max_seq_len,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
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

        # Reset the memory and weights
        self.flush()

        yield "Fine-tuning complete successfully!"


model_service = ModelService()

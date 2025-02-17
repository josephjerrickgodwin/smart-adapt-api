import gc
import logging
import os
import re
import torch

from threading import Thread
from typing import List
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
    AsyncTextIteratorStreamer
)
from trl import SFTTrainer

from src.exception.fine_tuning_disabled_error import FineTuningDisabledError
from src.exception.inference_disabled_error import InferenceDisabledError
from src.service import prompt_service
from src.service.TextStreamer import SmartAdaptTextStreamer
from src.service.storage_manager import storage_manager
from src.service.inference_model_service import inference_model_service

load_dotenv()

logger = logging.getLogger(__name__)

# Load the Sentence Transformer Model and the HF token
MODEL_ID = str(os.getenv('MODEL_ID'))
HF_TOKEN = str(os.getenv('HF_TOKEN'))


class ModelService:
    """
    Main service for LLM inference.
    """
    def __init__(self):
        self.token = HF_TOKEN
        self.data_model_name = 'adapters'
        self.logs_dir = 'logs'

        # Define model and tokenizer for reasoning
        self.tokenizer = None
        self.model = None

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

    def extract_new_query(self, text: str):
        """
        Extracts text between <new_query> and </new_query> tags.

        Parameters:
            text (str): The input string containing the tags.

        Returns:
            str: Extracted text or None if no match is found.
        """
        # Regex pattern with grouping to capture the text between the tags
        pattern = r"<new_query>(.*?)</new_query>"
        
        # Search for the pattern and extract the first group if found
        match = re.search(pattern, text)
        
        # Return the captured group or the original text if no match
        return match.group(1) if match else text

    async def _load_adapter(self, user_id: str, merge_and_unload: bool = False):
        """
        Get the inference pipeline for the model.

        Args:
            user_id (str): Unique ID of the user

        Returns:
            text-generation pipeline
        """
        # Check for the current user
        if not self.current_user:
            self.current_user = user_id

        logger.info('Started checking for LoRA adapters...')

        # If the user adapter is already exist, skip loading it again
        if self.lora_loaded:
            if user_id == self.current_user:
                logger.info('LoRA adapter already loaded. Skipping auto assign')
                return
            else:
                logger.info('Removing LoRA adapter of the previous user')
                self.model.delete_adapter(self.current_user)
                self.lora_loaded = False

        # Check for LoRA adapters for the current user in the local storage
        lora_exist = await storage_manager.check_file_exists(
            user_id=user_id,
            filename=self.data_model_name
        )
        logger.info(f"{'Found LoRA adapters and loading to the model' if lora_exist else 'Found no LoRA adapters'}")

        # Load the model with LoRA, if the weights are available
        if lora_exist:
            user_data_weights = await storage_manager.read(
                user_id=user_id,
                filename=self.data_model_name
            )
            peft_config = PeftConfig.from_pretrained(user_data_weights)
            self.model = PeftModel.from_pretrained(
                self.model,
                peft_config,
                adapter_name=user_id
            ).to(self.device)

            # Merge the adapter permanently
            if merge_and_unload:
                self.model = self.model.merge_and_unload()

            self.lora_loaded = True
            logger.info('Successfully loaded LoRA adapters')

        # Assign the current user
        self.current_user = user_id

    async def rewrite_query(
            self,
            current_query: str,
            history: list
    ):
        logger.info('Started extracting user messages from the history...')
        # Extract previous queries
        previous_queries_list = [
            f"- '{conversation.get('content', '')}'"
            for conversation in history
            if conversation.get('role', '') == 'user'
        ]
        previous_user_queries = '\n'.join(previous_queries_list)

        # Add the current query below the user previous user queries
        prompt = f'Previous Queries:\n{previous_user_queries}\nCurrent Query: *** {current_query} ***\nOutput: '

        logger.info('Preparing messages for prompting the LLM...')

        # Create a new history with the system prompt
        system_prompt = prompt_service.query_rewrite_system_instruction.strip()
        updated_history = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Apply chat template
        updated_messages = self.tokenizer.apply_chat_template(updated_history, tokenize=False)

        # Tokenize the messages
        tokens = self.tokenizer(
            updated_messages,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.device)

        # Define the streamer
        streamer = AsyncTextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Define the generation kwargs
        thinker_kwargs = dict(
            input_ids=tokens.input_ids,
            max_new_tokens=512,
            attention_mask=tokens.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer
        )
        logger.info('Started generating the updated query...')

        # Define the thread
        thread = Thread(target=self.model.generate, kwargs=thinker_kwargs)

        # Start the thread
        thread.start()

        # Start streaming the reasoning
        text = ''
        async for chunk in streamer:
            if chunk:
                text += chunk

        # Extract the new query
        new_query = self.extract_new_query(text)

        logger.info('Successfully generated the updated query.')

        return new_query

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
        
        logger.info(f'Context size: {len(context)}')

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

        reasoned_context = ''
        if context:
            logger.info('Started re-arranging the message for reasoning')

            # Format messages
            system_prompt_for_cot = prompt_service.system_prompt_for_thinker_model.strip()

            # Add context to reasoning
            updated_query = f'<information>\n{context}\n</information>\n{user_query}' if context else user_query

            # Define the conversation
            messages = [
                {
                    "role": "system",
                    "content": system_prompt_for_cot
                },
                {
                    "role": "user",
                    "content": updated_query.strip()
                }
            ]

            # Apply chat template
            updated_messages = self.tokenizer.apply_chat_template(messages, tokenize=False)

            logger.info('Started tokenizing the conversation')

            # Tokenize the messages
            tokens = self.tokenizer(
                updated_messages,
                add_special_tokens=True,
                return_tensors="pt"
            ).to(self.device)

            logger.info('Started defining the hyperparameters for reasoning')

            # Define the generation kwargs
            thinker_kwargs = dict(
                input_ids=tokens.input_ids,
                max_new_tokens=512,
                attention_mask=tokens.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                # do_sample=True,
                # repetition_penalty=1.8,
                streamer=streamer
            )

            logger.info('Started the reasoning process')

            # Define the thread
            thread = Thread(target=self.model.generate, kwargs=thinker_kwargs)

            # Start the thread
            thread.start()

            # # Start streaming the reasoning
            # async for chunk in streamer:
            #     if chunk:
            #         yield f'data: {chunk}\n\n'

            logger.info('Extracting the reasining and preparing for inference')

            # Get the streamed message from the `thinker` model
            reasoning = streamer.get_response()

            # Add `context` and reasoning as a new context
            reasoned_context = f'<information>\n{context}\n\nExplanation:\n{reasoning}\n</information>'

        # Update the streamer for `text` streaming
        streamer.update_text_type('text')
        logger.info('Starting the inference')

        # Start inference
        async for chunk in inference_model_service.get_result(
            query=user_query,
            context=reasoned_context,
            history=history
        ):
            yield chunk

        logger.info('Request have been served successfully')

    def _load_weights(self):
        """
        Loads the `thinker` and `inference` model instances
        """
        # Load the tokenizer
        logger.info('Started loading the tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=self.token
        )

        # Load the model
        logger.info('Started loading the model')
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=self.token,
            device_map=self.device
        )

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
    ):
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
        if not self.inference_enabled:
            raise FineTuningDisabledError('A fine-tuning task is in progress!')

        # Disable inference during training
        self.inference_enabled = False

        # Unload and reset weights
        self.flush()

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

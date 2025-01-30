import time
from typing import Literal

from transformers import AsyncTextIteratorStreamer, AutoTokenizer

from src.model.llm_response_model import DeltaModel, ChoicesModel, LLMResponseModel


class SmartAdaptTextStreamer(AsyncTextIteratorStreamer):
    def __init__(
            self,
            tokenizer: "AutoTokenizer",
            text_type: Literal["thinking", "text"] = "thinking",
            model: str = '',
            index: int = 0,
            message_id: int = 0,
            parent_id: int = 0,
            **decode_kwargs,
    ):
        super().__init__(tokenizer, **decode_kwargs)
        self.text_type = text_type
        self.created_at = int(time.time())

        # Attributes to update during the completion
        self.model = model
        self.index = index
        self.chunk_token_usage = 0
        self.message_id = message_id
        self.parent_id = parent_id

        # Save the generated text
        self.response = ''

    def update_text_type(self, text_type: Literal["thinking", "text"]):
        self.text_type = text_type

        # Reset the request creation time and the previous response
        self.created_at = int(time.time())
        self.response = ''

    def get_response(self):
        return self.response

    def put(self, value):
        # Count tokens added in this step
        num_tokens = len(value.tolist())
        self.chunk_token_usage += num_tokens
        super().put(value)

    def on_finalized_text(
            self,
            text: str,
            stream_end: bool = False
    ):
        # Calculate the time taken
        current_time = int(time.time())
        time_taken = current_time - self.created_at

        # Clean and update the response
        filtered_text = text.replace("assistant\n\n", "").strip()

        response = ''
        if filtered_text:
            self.response += filtered_text

            # Construct the event
            delta = DeltaModel(content=filtered_text, type=self.text_type)
            choices = ChoicesModel(index=self.index, delta=delta)
            response = LLMResponseModel(
                choices=choices,
                model=self.model,
                chunk_token_usage=self.chunk_token_usage,
                created=self.created_at,
                message_id=self.message_id,
                parent_id=self.parent_id,
                time_elapsed=time_taken
            ).to_dict()

            # Convert to string
            response = f'{response}'

        # Put the new text in the queue. If the stream is ending, also put a stop signal in the queue.
        super().on_finalized_text(
            text=response,
            stream_end=stream_end
        )

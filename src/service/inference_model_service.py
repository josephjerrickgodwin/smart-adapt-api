import os
import logging
import aiohttp
import json

from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

logger = logging.getLogger(__name__)

# Load the Sentence Transformer Model and the HF token
INFERENCE_NODE = str(os.getenv('INFERENCE_NODE'))


class InferenceModelService:
    def __init__(self):
        self.inference_node = f'{INFERENCE_NODE}/v1/completions'
        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'Connection': 'keep-alive'
        }

    async def get_result(self, query: str, context: str, history: List[Dict[str, Any]]):
        # Prepare the request body
        request_body = {
            'query': query,
            'history': history,
            'context': context
        }

        # Send the payload
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=self.inference_node,
                headers=self.headers,
                json=request_body
            ) as response:
                if response.status == 200:
                    # Stream the response content in chunks
                    async for chunk in response.content.iter_chunked(1024):
                        chunk = chunk.decode('utf-8')
                        chunk = chunk.replace('data: ', '')
                        yield chunk
                else:
                    response.raise_for_status()


inference_model_service = InferenceModelService()

import logging
import os
from io import BytesIO
from typing import Literal

import aiohttp
from dotenv import load_dotenv

from src.service.llm.hf_client import hf_client

load_dotenv()

logger = logging.getLogger(__name__)

# Load the Sentence Transformer Model and the HF token
CLIENT_NODE = str(os.getenv('CLIENT_NODE'))


class ClientService:
    def __init__(self):
        self.chat_completions_endpoint = f'{CLIENT_NODE}/api/v1/completions'
        self.query_rewrite_endpoint = f'{CLIENT_NODE}/api/v1/rewrite'
        self.fine_tuning_endpoint = f'{CLIENT_NODE}/api/v1/fine-tune'
        self.validate_user_knowledge = f'{CLIENT_NODE}/api/v1/validate'
        self.get_knowledge_data_endpoint = f'{CLIENT_NODE}/api/v1//adapter/download'
        self.delete_lora_adapter_endpoint = f'{CLIENT_NODE}/api/v1/adapter'
        self.stop_fine_tuning_endpoint = f'{CLIENT_NODE}/api/v1/fine-tune/stop'

        self.default_header = {
            'Connection': 'keep-alive'
        }
        self.json_headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

    async def get_knowledge_data_using_client(
            self,
            user_role: Literal['admin', 'user'],
            knowledge_ids=None
    ):
        if knowledge_ids is None:
            knowledge_ids = []

        request_body = {
            'user_role': user_role,
            'knowledge_ids': knowledge_ids
        }

        # Send the payload
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    url=self.validate_user_knowledge,
                    headers={
                        **self.default_header,
                        **self.json_headers
                    },
                    json=request_body,
                    timeout=None
            ) as response:
                if response.status == 200:
                    output = await response.json()
                    return output.get('knowledge_bases', [])
                else:
                    response.raise_for_status()

    async def rewrite_query_using_client(self, query: str, history: list):
        request_body = {
            'query': query,
            'history': history
        }

        # Send the payload
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    url=self.query_rewrite_endpoint,
                    headers={
                        **self.default_header,
                        **self.json_headers
                    },
                    params=request_body,
                    timeout=None
            ) as response:
                if response.status == 200:
                    output = await response.json()
                    return output.get('query', query)
                else:
                    response.raise_for_status()

    async def start_completions_using_client(
            self,
            user_id: str,
            messages: list,
            stream: bool,
            knowledge_ids: list = None
    ):
        request_body = {
            "user_id": user_id,
            "stream": str(stream),
            'messages': messages,
        }
        if knowledge_ids:
            request_body['knowledge_ids'] = knowledge_ids

        # Send the payload
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    url=self.chat_completions_endpoint,
                    headers={
                        **self.default_header,
                        **self.json_headers
                    },
                    json=request_body,
                    timeout=None
            ) as response:
                if response.status == 200:
                    async for chunk in response.content.iter_chunked(1024):
                        chunk = chunk.decode('utf-8', errors='ignore')
                        yield chunk
                else:
                    response.raise_for_status()

    async def fine_tuning_using_client(
            self,
            user_id: str,
            knowledge_id: str,
            question_column_name: str,
            answer_column_name: str,
            file_stream: BytesIO
    ):
        # Prepare form data to send as multipart/form-data
        form_data = aiohttp.FormData()
        form_data.add_field(
            'file',
            file_stream,
            filename='data.csv',
            content_type='text/csv'
        )
        form_data.add_field('user_id', user_id)
        form_data.add_field('knowledge_id', knowledge_id)
        form_data.add_field('question_column_name', question_column_name)
        form_data.add_field('answer_column_name', answer_column_name)

        # Send the payload
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    url=self.fine_tuning_endpoint,
                    data=form_data,
                    timeout=None
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    response.raise_for_status()

    async def remove_lora_adapter_using_client(self, user_id: str, knowledge_id: str):
        request_params = {
            "user_id": user_id,
            "knowledge_id": knowledge_id
        }

        # Send the payload
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                    url=self.delete_lora_adapter_endpoint,
                    headers={
                        **self.default_header,
                        **self.json_headers
                    },
                    params=request_params,
                    timeout=None
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    return None
                else:
                    response.raise_for_status()

    async def download_lora_adapter_using_client(self, user_id: str, knowledge_id: str):
        """Return an async generator that streams the LoRA adapter bytes.

        We wrap the request logic inside the generator so the HTTP connection
        remains open for the entire duration of the streaming, avoiding the
        ClientConnectionError that occurs when the response context manager
        exits too early.
        """

        request_params = {
            "user_id": user_id,
            "knowledge_id": knowledge_id,
        }

        async def stream():
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url=self.get_knowledge_data_endpoint,
                    headers=self.default_header,
                    params=request_params,
                    timeout=None,
                ) as response:
                    if response.status != 200:
                        response.raise_for_status()

                    async for chunk in response.content.iter_chunked(1024):
                        yield chunk

        return stream()

    async def stop_fine_tuning_using_client(self, user_id: str, knowledge_id: str):
        request_body = {
            "user_id": user_id,
            "knowledge_id": knowledge_id
        }

        # aiohttp requires data to be a dict for x-www-form-urlencoded
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=self.stop_fine_tuning_endpoint,
                headers={
                    **self.default_header,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data=request_body,
                timeout=None
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    response.raise_for_status()


client_service = ClientService()

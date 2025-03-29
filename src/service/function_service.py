import inspect
import json
import logging
import sys
from typing import AsyncGenerator, Generator, Iterator

from fastapi import (
    Request,
)
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from src.model.functions import Functions
from src.model.models import Models
from src.service.env import SRC_LOG_LEVELS, GLOBAL_LOG_LEVEL
from src.service.fine_tuning.model_service import model_service
from src.service.sockets import (
    get_event_call,
    get_event_emitter,
)
from src.service.utils.misc_service import (
    openai_chat_chunk_message_template,
    openai_chat_completion_message_template,
)
from src.service.utils.payload_service import (
    apply_model_params_to_body_openai,
    apply_model_system_prompt_to_body,
)
from src.service.utils.plugin_service import load_function_module_by_id
from src.service.utils.tool_service import get_tools

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


def get_function_module_by_id(request: Request, pipe_id: str):
    # Check if function is already loaded
    if pipe_id not in request.app.state.FUNCTIONS:
        function_module, _, _ = load_function_module_by_id(pipe_id)
        request.app.state.FUNCTIONS[pipe_id] = function_module
    else:
        function_module = request.app.state.FUNCTIONS[pipe_id]

    if hasattr(function_module, "valves") and hasattr(function_module, "Valves"):
        valves = Functions.get_function_valves_by_id(pipe_id)
        function_module.valves = function_module.Valves(**(valves if valves else {}))
    return function_module


async def get_function_models(request):
    pipes = Functions.get_functions_by_type("pipe", active_only=True)
    pipe_models = []
    for pipe in pipes:
        function_module = get_function_module_by_id(request, pipe.id)

        # Check if function is a manifold
        if hasattr(function_module, "pipes"):
            # Check if pipes is a function or a list
            try:
                if callable(function_module.pipes):
                    sub_pipes = function_module.pipes()
                else:
                    sub_pipes = function_module.pipes
            except Exception as e:
                log.exception(e)
                sub_pipes = []

            log.debug(
                f"get_function_models: function '{pipe.id}' is a manifold of {sub_pipes}"
            )

            for p in sub_pipes:
                sub_pipe_id = f'{pipe.id}.{p["id"]}'
                sub_pipe_name = p["name"]

                if hasattr(function_module, "name"):
                    sub_pipe_name = f"{function_module.name}{sub_pipe_name}"

                pipe_models.append(
                    {
                        "id": sub_pipe_id,
                        "name": sub_pipe_name,
                        "object": "model",
                        "created": pipe.created_at,
                        "owned_by": "openai",
                        "pipe": {"type": pipe.type},
                    }
                )
        else:
            log.debug(
                f"get_function_models: function '{pipe.id}' is a single pipe {{ 'id': {pipe.id}, 'name': {pipe.name} }}"
            )
            pipe_models.append(
                {
                    "id": pipe.id,
                    "name": pipe.name,
                    "object": "model",
                    "created": pipe.created_at,
                    "owned_by": "openai",
                    "pipe": {"type": pipe.type},
                }
            )

    return pipe_models


async def generate_function_chat_completion(
        form_data, user, stream
):
    metadata = form_data.pop("metadata", {})

    # Extract 'params' dictionary from form_data, defaulting to an empty dictionary if missing
    params = form_data.pop("params", {})

    # form_data = apply_model_params_to_body_openai(params, form_data)
    form_data = apply_model_system_prompt_to_body(
        params=params,
        form_data=form_data,
        metadata=metadata,
        user=user
    )

    async def stream_content():
        try:
            # Define the knowledge sources
            files = metadata.get('files', [])
            knowledge_sources = [
                file_info.get('id')
                for file_info in files
                if file_info.get('data', {}).get('status', '') == 'Completed'
            ]
            async for chunk in model_service.test_completions(
                    user_id=user.id,
                    messages=form_data['messages'],
                    stream=stream,
                    knowledge_ids=knowledge_sources,
                    **params
            ):
                if isinstance(chunk, str):
                    message = openai_chat_chunk_message_template(form_data["model"], chunk)
                    yield f"data: {json.dumps(message)}\n\n"

                elif chunk is None:
                    finish_message = openai_chat_chunk_message_template(
                        form_data["model"], ""
                    )
                    finish_message["choices"][0]["finish_reason"] = "stop"
                    yield f"data: {json.dumps(finish_message)}\n\n"
                    yield "data: [DONE]"

        except Exception as e:
            log.error(f"Error: {e}")
            yield f"data: {json.dumps({'error': {'detail': str(e)}})}\n\n"
            return

    return StreamingResponse(stream_content(), media_type="text/event-stream")

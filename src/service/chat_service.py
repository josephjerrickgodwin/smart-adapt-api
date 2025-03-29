import asyncio
import asyncio
import inspect
import json
import logging
import random
import sys
import uuid
from typing import Any

from fastapi import Request
from starlette.responses import StreamingResponse

from src.controller.chat_ui.model_controller import get_model_by_id
from src.model.functions import Functions
from src.service.env import SRC_LOG_LEVELS, GLOBAL_LOG_LEVEL, BYPASS_MODEL_ACCESS_CONTROL
from src.service.function_service import generate_function_chat_completion
from src.service.sockets import (
    sio,
    get_event_call,
    get_event_emitter,
)
from src.service.utils.filter_service import (
    get_sorted_filter_ids,
    process_filter_functions,
)
from src.service.utils.model_service import get_all_models, check_model_access
from src.service.utils.pipeline_service import (
    process_pipeline_outlet_filter,
)
from src.service.utils.plugin_service import load_function_module_by_id

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


async def generate_direct_chat_completion(
    request: Request,
    form_data: dict,
    user: Any,
    models: dict,
):
    print("generate_direct_chat_completion")

    metadata = form_data.pop("metadata", {})

    user_id = metadata.get("user_id")
    session_id = metadata.get("session_id")
    request_id = str(uuid.uuid4())  # Generate a unique request ID

    event_caller = get_event_call(metadata)

    channel = f"{user_id}:{session_id}:{request_id}"

    if form_data.get("stream"):
        q = asyncio.Queue()

        async def message_listener(sid, data):
            """
            Handle received socket messages and push them into the queue.
            """
            await q.put(data)

        # Register the listener
        sio.on(channel, message_listener)

        # Start processing chat completion in background
        res = await event_caller(
            {
                "type": "request:chat:completion",
                "data": {
                    "form_data": form_data,
                    "model": models[form_data["model"]],
                    "channel": channel,
                    "session_id": session_id,
                },
            }
        )

        print("res", res)

        if res.get("status", False):
            # Define a generator to stream responses
            async def event_generator():
                nonlocal q
                try:
                    while True:
                        data = await q.get()  # Wait for new messages
                        if isinstance(data, dict):
                            if "done" in data and data["done"]:
                                break  # Stop streaming when 'done' is received

                            yield f"data: {json.dumps(data)}\n\n"
                        elif isinstance(data, str):
                            yield data
                except Exception as e:
                    log.debug(f"Error in event generator: {e}")
                    pass

            # Define a background task to run the event generator
            async def background():
                try:
                    del sio.handlers["/"][channel]
                except Exception as e:
                    pass

            # Return the streaming response
            return StreamingResponse(
                event_generator(), media_type="text/event-stream", background=background
            )
        else:
            raise Exception(str(res))
    else:
        res = await event_caller(
            {
                "type": "request:chat:completion",
                "data": {
                    "form_data": form_data,
                    "model": models[form_data["model"]],
                    "channel": channel,
                    "session_id": session_id,
                },
            }
        )

        if "error" in res:
            raise Exception(res["error"])

        return res


async def generate_chat_completion(
        request: Request,
        form_data: dict,
        user: Any,
        bypass_filter: bool = True,
        stream: bool = True
):
    # if BYPASS_MODEL_ACCESS_CONTROL:
    #     bypass_filter = True

    model_id = form_data["model"]
    model = await get_model_by_id(id=model_id, user=user)

    # # Check if user has access to the model
    # if not bypass_filter and user.role == "user":
    #     check_model_access(user, model)

    return await generate_function_chat_completion(
        form_data=form_data,
        user=user,
        stream=stream
    )


# chat_completion = generate_chat_completion


async def chat_completed(request: Request, form_data: dict, user: Any):
    if not request.app.state.MODELS:
        await get_all_models(request)

    data = form_data
    model_id = data["model"]
    model = await get_model_by_id(id=model_id, user=user)

    metadata = {
        "chat_id": data["chat_id"],
        "message_id": data["id"],
        "session_id": data["session_id"],
        "user_id": user.id,
    }

    extra_params = {
        "__event_emitter__": get_event_emitter(metadata),
        "__event_call__": get_event_call(metadata),
        "__user__": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
        },
        "__metadata__": metadata,
        "__request__": request,
        "__model__": model,
    }

    try:
        result, _ = await process_filter_functions(
            request=request,
            filter_ids=get_sorted_filter_ids(model),
            filter_type="outlet",
            form_data=data,
            extra_params=extra_params,
        )
        return result
    except Exception as e:
        return Exception(f"Error: {e}")


async def chat_action(request: Request, action_id: str, form_data: dict, user: Any):
    if "." in action_id:
        action_id, sub_action_id = action_id.split(".")
    else:
        sub_action_id = None

    action = Functions.get_function_by_id(action_id)
    if not action:
        raise Exception(f"Action not found: {action_id}")

    if not request.app.state.MODELS:
        await get_all_models(request)

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    data = form_data
    model_id = data["model"]

    if model_id not in models:
        raise Exception("Model not found")
    model = models[model_id]

    __event_emitter__ = get_event_emitter(
        {
            "chat_id": data["chat_id"],
            "message_id": data["id"],
            "session_id": data["session_id"],
            "user_id": user.id,
        }
    )
    __event_call__ = get_event_call(
        {
            "chat_id": data["chat_id"],
            "message_id": data["id"],
            "session_id": data["session_id"],
            "user_id": user.id,
        }
    )

    if action_id in request.app.state.FUNCTIONS:
        function_module = request.app.state.FUNCTIONS[action_id]
    else:
        function_module, _, _ = load_function_module_by_id(action_id)
        request.app.state.FUNCTIONS[action_id] = function_module

    if hasattr(function_module, "valves") and hasattr(function_module, "Valves"):
        valves = Functions.get_function_valves_by_id(action_id)
        function_module.valves = function_module.Valves(**(valves if valves else {}))

    if hasattr(function_module, "action"):
        try:
            action = function_module.action

            # Get the signature of the function
            sig = inspect.signature(action)
            params = {"body": data}

            # Extra parameters to be passed to the function
            extra_params = {
                "__model__": model,
                "__id__": sub_action_id if sub_action_id is not None else action_id,
                "__event_emitter__": __event_emitter__,
                "__event_call__": __event_call__,
                "__request__": request,
            }

            # Add extra params in contained in function signature
            for key, value in extra_params.items():
                if key in sig.parameters:
                    params[key] = value

            if "__user__" in sig.parameters:
                __user__ = {
                    "id": user.id,
                    "email": user.email,
                    "name": user.name,
                    "role": user.role,
                }

                try:
                    if hasattr(function_module, "UserValves"):
                        __user__["valves"] = function_module.UserValves(
                            **Functions.get_user_valves_by_id_and_user_id(
                                action_id, user.id
                            )
                        )
                except Exception as e:
                    print(e)

                params = {**params, "__user__": __user__}

            if inspect.iscoroutinefunction(action):
                data = await action(**params)
            else:
                data = action(**params)

        except Exception as e:
            return Exception(f"Error: {e}")

    return data

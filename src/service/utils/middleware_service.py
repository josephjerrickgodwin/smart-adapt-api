import ast
import asyncio
import html
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from uuid import uuid4

from fastapi import Request
from starlette.responses import StreamingResponse

from src.controller.chat_ui.task_controller import (
    generate_queries,
    generate_title,
)
from src.model.chats import Chats
from src.model.constants import TASKS
from src.model.users import UserModel
from src.model.users import Users
from src.service.agentic_rag_service import AgenticRAGService
from src.service.chat_service import generate_chat_completion
from src.service.env import (
    SRC_LOG_LEVELS,
    GLOBAL_LOG_LEVEL,
    ENABLE_REALTIME_CHAT_SAVE,
)
from src.service.sockets import (
    get_event_call,
    get_event_emitter,
    get_active_status_by_user_id,
)
from src.service.utils.code_interpreter_service import execute_code_jupyter
from src.service.utils.config_service import (
    DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE,
    CODE_INTERPRETER_ENGINE, DEFAULT_RAG_TEMPLATE, )
from src.service.utils.misc_service import (
    get_message_list,
    add_or_update_system_message,
    get_last_user_message,
    get_last_assistant_message, add_user_message,
)
from src.service.utils.storage.storage_service import Storage
from src.service.utils.storage.util_service import get_sources_from_files
from src.service.utils.task_service import (
    create_task,
    get_task_model_id,
    tools_function_calling_generation_template,
)
from src.service.utils.webhook import post_webhook

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


async def chat_completion_tools_handler(
        request: Request, body: dict, user: UserModel, models, tools
) -> tuple[dict, dict]:
    async def get_content_from_response(response) -> Optional[str]:
        content = None
        if hasattr(response, "body_iterator"):
            async for chunk in response.body_iterator:
                data = json.loads(chunk.decode("utf-8"))
                content = data["choices"][0]["message"]["content"]

            # Cleanup any remaining background tasks if necessary
            if response.background is not None:
                await response.background()
        else:
            content = response["choices"][0]["message"]["content"]
        return content

    def get_tools_function_calling_payload(messages, task_model_id, content):
        user_message = get_last_user_message(messages)
        history = "\n".join(
            f"{message['role'].upper()}: \"\"\"{message['content']}\"\"\""
            for message in messages[::-1][:4]
        )

        prompt = f"History:\n{history}\nQuery: {user_message}"

        return {
            "model": task_model_id,
            "messages": [
                {"role": "system", "content": content},
                {"role": "user", "content": f"Query: {prompt}"},
            ],
            "stream": False,
            "metadata": {"task": str(TASKS.FUNCTION_CALLING)},
        }

    task_model_id = get_task_model_id(
        body["model"],
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )

    skip_files = False
    sources = []

    specs = [tool["spec"] for tool in tools.values()]
    tools_specs = json.dumps(specs)

    if request.app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE != "":
        template = request.app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE
    else:
        template = DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE

    tools_function_calling_prompt = tools_function_calling_generation_template(
        template, tools_specs
    )
    log.info(f"{tools_function_calling_prompt=}")
    payload = get_tools_function_calling_payload(
        body["messages"], task_model_id, tools_function_calling_prompt
    )

    try:
        response = await generate_chat_completion(request, form_data=payload, user=user)
        log.debug(f"{response}")
        content = await get_content_from_response(response)
        log.debug(f"{content}")

        if not content:
            return body, {}

        try:
            content = content[content.find("{"): content.rfind("}") + 1]
            if not content:
                raise Exception("No JSON object found in the response")

            result = json.loads(content)

            async def tool_call_handler(tool_call):
                nonlocal skip_files

                log.debug(f"{tool_call=}")

                tool_function_name = tool_call.get("name", None)
                if tool_function_name not in tools:
                    return body, {}

                tool_function_params = tool_call.get("parameters", {})

                try:
                    required_params = (
                        tools[tool_function_name]
                        .get("spec", {})
                        .get("parameters", {})
                        .get("required", [])
                    )
                    tool_function = tools[tool_function_name]["callable"]
                    tool_function_params = {
                        k: v
                        for k, v in tool_function_params.items()
                        if k in required_params
                    }
                    tool_output = await tool_function(**tool_function_params)

                except Exception as e:
                    tool_output = str(e)

                if isinstance(tool_output, str):
                    if tools[tool_function_name]["citation"]:
                        sources.append(
                            {
                                "source": {
                                    "name": f"TOOL:{tools[tool_function_name]['toolkit_id']}/{tool_function_name}"
                                },
                                "document": [tool_output],
                                "metadata": [
                                    {
                                        "source": f"TOOL:{tools[tool_function_name]['toolkit_id']}/{tool_function_name}"
                                    }
                                ],
                            }
                        )
                    else:
                        sources.append(
                            {
                                "source": {},
                                "document": [tool_output],
                                "metadata": [
                                    {
                                        "source": f"TOOL:{tools[tool_function_name]['toolkit_id']}/{tool_function_name}"
                                    }
                                ],
                            }
                        )

                    if tools[tool_function_name]["file_handler"]:
                        skip_files = True

            # check if "tool_calls" in result
            if result.get("tool_calls"):
                for tool_call in result.get("tool_calls"):
                    await tool_call_handler(tool_call)
            else:
                await tool_call_handler(result)

        except Exception as e:
            log.exception(f"Error: {e}")
            content = None
    except Exception as e:
        log.exception(f"Error: {e}")
        content = None

    log.debug(f"tool_contexts: {sources}")

    if skip_files and "files" in body.get("metadata", {}):
        del body["metadata"]["files"]

    return body, {"sources": sources}


async def chat_web_search_handler(
        request: Request, form_data: dict, extra_params: dict, user
):
    event_emitter = extra_params["__event_emitter__"]
    await event_emitter(
        {
            "type": "status",
            "data": {
                "action": "web_search",
                "description": "Generating search query",
                "done": False,
            },
        }
    )

    messages = form_data["messages"]
    user_message = get_last_user_message(messages)

    queries = []
    try:
        res = await generate_queries(
            request,
            {
                "model": form_data["model"],
                "messages": messages,
                "prompt": user_message,
                "type": "web_search",
            },
            user,
        )

        response = res["choices"][0]["message"]["content"]

        try:
            bracket_start = response.find("{")
            bracket_end = response.rfind("}") + 1

            if bracket_start == -1 or bracket_end == -1:
                raise Exception("No JSON object found in the response")

            response = response[bracket_start:bracket_end]
            queries = json.loads(response)
            queries = queries.get("queries", [])
        except Exception as e:
            queries = [response]

    except Exception as e:
        log.exception(e)
        queries = [user_message]

    if len(queries) == 0:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "action": "web_search",
                    "description": "No search query generated",
                    "done": True,
                },
            }
        )
        return form_data

    searchQuery = queries[0]

    await event_emitter(
        {
            "type": "status",
            "data": {
                "action": "web_search",
                "description": 'Searching "{{searchQuery}}"',
                "query": searchQuery,
                "done": False,
            },
        }
    )
    await event_emitter(
        {
            "type": "status",
            "data": {
                "action": "web_search",
                "description": "No search results found",
                "query": searchQuery,
                "done": True,
                "error": True,
            },
        }
    )
    return form_data


async def chat_completion_files_handler(
        request: Request, body: dict, user: UserModel
) -> tuple[dict, dict[str, list]]:
    sources = []

    if files := body.get("metadata", {}).get("files", None):
        try:
            queries_response = await generate_queries(
                request,
                {
                    "model": body["model"],
                    "messages": body["messages"],
                    "type": "retrieval",
                },
                user,
            )
            queries_response = queries_response["choices"][0]["message"]["content"]

            try:
                bracket_start = queries_response.find("{")
                bracket_end = queries_response.rfind("}") + 1

                if bracket_start == -1 or bracket_end == -1:
                    raise Exception("No JSON object found in the response")

                queries_response = queries_response[bracket_start:bracket_end]
                queries_response = json.loads(queries_response)
            except Exception as e:
                queries_response = {"queries": [queries_response]}

            queries = queries_response.get("queries", [])
        except Exception as e:
            queries = []

        if len(queries) == 0:
            queries = [get_last_user_message(body["messages"])]

        try:
            # Offload get_sources_from_files to a separate thread
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as executor:
                sources = await loop.run_in_executor(
                    executor,
                    lambda: get_sources_from_files(
                        files=files,
                        queries=queries,
                        embedding_function=lambda query: request.app.state.EMBEDDING_FUNCTION(
                            query, user=user
                        ),
                        k=request.app.state.config.TOP_K,
                        reranking_function=request.app.state.rf,
                        r=request.app.state.config.RELEVANCE_THRESHOLD,
                        hybrid_search=request.app.state.config.ENABLE_RAG_HYBRID_SEARCH,
                    ),
                )

        except Exception as e:
            log.exception(e)

        log.debug(f"rag_contexts:sources: {sources}")

    return body, {"sources": sources}


def apply_params_to_form_data(form_data: dict):
    """
    Extracts specific parameters from the 'params' key in the given form_data dictionary
    and applies them directly to the main form_data dictionary.

    Args:
        form_data (dict): A dictionary containing form data, including an optional 'params' key
                          with additional configuration parameters.

    Returns:
        dict: The updated form_data dictionary with extracted parameters applied.
    """
    # Extract 'params' dictionary from form_data, defaulting to an empty dictionary if missing
    params = form_data.pop("params", {})

    # Apply specific parameters if they exist in 'params'
    if "temperature" in params:
        form_data["temperature"] = params["temperature"]

    if "max_tokens" in params:
        form_data["max_tokens"] = params["max_tokens"]

    if "top_p" in params:
        form_data["top_p"] = params["top_p"]

    if "frequency_penalty" in params:
        form_data["frequency_penalty"] = params["frequency_penalty"]

    return form_data


def extract_new_query(text: str):
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


async def process_chat_payload(form_data: dict, metadata, user, event_emitter=None):
    events = []
    context = ''

    # Extract history from the form data
    history = form_data["messages"]
    user_message = get_last_user_message(history)

    # If custom knowledge is given, refrain from using agentic RAG
    files = metadata.get('files', [])
    files = files if files else []
    knowledge_count, files_count = 0, 0
    for file_info in files:
        file_type = file_info.get('type', '')
        if file_type == 'file':
            files_count += 1
        elif file_type == 'collection':
            knowledge_count += 1

    # Based on the file types, determine the retrieval
    use_agentic_rag = True
    if knowledge_count and files_count:
        use_agentic_rag = True
    elif knowledge_count:
        use_agentic_rag = False

    rag_service = None
    if use_agentic_rag:
        try:
            # Get the index file from the DB
            log.info("Started fetching the existing index for agentic RAG")

            # Format the index file name
            index_filename = f'{user.id}__index.pkl'

            # Check if an index is available for the user
            rag_service = Storage.get_file(index_filename)
            if rag_service:
                log.info("Initializing agentic RAG service")
                agentic_rag_service = AgenticRAGService(rag_service)

                # Process with agentic RAG, pass event_emitter
                use_rag, memory = await agentic_rag_service.process_agentic_rag(
                    history=history,
                    query=user_message or "",
                    event_emitter=event_emitter
                )
                if use_rag and memory:
                    context = memory.context.strip()
                    log.info(
                        f'Agentic RAG retrieved {len(memory.sources)} context items across '
                        f'{memory.sources[-1]["iteration"] if memory.sources else 0} iterations'
                    )
                else:
                    log.info("Agentic RAG determined RAG was not needed for this query")
            else:
                log.warning("No RAG index found for agentic RAG processing")

        except Exception as e:
            log.error(f"Error in agentic RAG processing: {e}")

            # Fallback to traditional RAG if agentic RAG fails
            log.info("Falling back to traditional RAG")
            try:
                if rag_service:
                    sources = await rag_service.search(query=user_message)
                    for source in sources:
                        context += f"\n{source['label']}".strip()
                    log.info(f'Traditional RAG fallback retrieved {len(sources)} context items')
            except Exception as fallback_error:
                log.error(f"Traditional RAG fallback also failed: {fallback_error}")

    if event_emitter:
        await event_emitter({
            "type": "status",
            "data": {
                "action": "thinking",
                "description": f"Finalizing the result",
                "done": True
            }
        })

    # Always ensure the system message is present and up to date
    history = add_or_update_system_message(
        content=DEFAULT_RAG_TEMPLATE,
        messages=history,
    )

    # Implement the user message (with or without context)
    history = add_user_message(
        content=user_message or "",
        messages=history,
        context=context.strip() if context else ""
    )

    # Update messages
    form_data["messages"] = history

    return form_data, metadata, events


def extract_title_from_text(text):
    # Use regex to find the dictionary in the text
    match = re.search(r'\{.*}', text)
    if match:
        dict_str = match.group(0)

        # Convert the string to a dictionary using ast.literal_eval for safety
        extracted_dict = ast.literal_eval(dict_str)

        # Extract the title from the dictionary, default to "New Chat" if not found
        return extracted_dict.get('title', "New Chat")

    return "New Chat"


async def process_chat_response(
        request, response, form_data, user, events, metadata, tasks, event_emitter=None
):
    async def background_tasks_handler():
        message_map = Chats.get_messages_by_chat_id(metadata["chat_id"])
        message = message_map.get(metadata["message_id"]) if message_map else None
        if not message:
            return

        messages = get_message_list(message_map, message.get("id"))
        if not tasks or not messages:
            return

        if TASKS.TITLE_GENERATION in tasks:
            if tasks[TASKS.TITLE_GENERATION]:
                res = await generate_title(
                    request,
                    {
                        "model": message["model"],
                        "messages": messages,
                        "chat_id": metadata["chat_id"],
                    },
                    user,
                )

                title = "New Chat"
                async for line in res.body_iterator:
                    line = line.decode("utf-8") if isinstance(line, bytes) else line

                    # Skip empty lines and events that are not formatted
                    if not line.strip() or not line.startswith("data:"):
                        continue
                    _data = line

                    # Remove the prefix
                    _data = _data[len("data:"):].strip()

                    # Check for the DONE token
                    if _data == '[DONE]':
                        continue

                    try:
                        _data = json.loads(_data)
                        choices = _data.get("choices", [])
                        if not choices:
                            continue

                        value = choices[0].get("delta", {}).get("content", '')
                        if not value:
                            continue

                        # Extract the chat title
                        title = extract_title_from_text(value)

                    except Exception as ex:
                        log.error(f'Title generation exception: {str(ex)}')

                Chats.update_chat_title_by_id(metadata["chat_id"], title)
                await event_emitter({"type": "chat:title", "data": title})

            elif len(messages) == 2:
                title = messages[0].get("content", "New Chat")
                Chats.update_chat_title_by_id(metadata["chat_id"], title)
                await event_emitter({"type": "chat:title", "data": message.get("content", "New Chat")})

        # if TASKS.TAGS_GENERATION in tasks and tasks[TASKS.TAGS_GENERATION]:
        #     res = await generate_chat_tags(
        #         request,
        #         {
        #             "model": message["model"],
        #             "messages": messages,
        #             "chat_id": metadata["chat_id"],
        #         },
        #         user,
        #     )
        #
        #     tags = ""
        #     async for line in res.body_iterator:
        #         line = line.decode("utf-8") if isinstance(line, bytes) else line
        #
        #         # Skip empty lines and events that are not formatted
        #         if not line.strip() or not line.startswith("data:"):
        #             continue
        #         _data = line
        #
        #         # Remove the prefix
        #         _data = _data[len("data:"):].strip()
        #
        #         try:
        #             _data = json.loads(_data)
        #             choices = _data.get("choices", [])
        #             if not choices:
        #                 continue
        #
        #             value = choices[0].get("delta", {}).get("content", '')
        #             if not value:
        #                 continue
        #
        #             # Update the title
        #             tags = value[value.find("{"): value.rfind("}") + 1]
        #         except Exception as ex:
        #             log.error(f'Tag generation exception: {str(ex)}')
        #
        #     Chats.update_chat_tags_by_id(metadata["chat_id"], tags, user)
        #     await event_emitter({"type": "chat:tags", "data": tags})

    def split_content_and_whitespace(content):
        content_stripped = content.rstrip()
        original_whitespace = content[len(content_stripped):] if len(content) > len(content_stripped) else ""
        return content_stripped, original_whitespace

    def is_opening_code_block(content):
        backtick_segments = content.split("```")
        # Even number of segments means the last backticks are opening a new block
        return len(backtick_segments) > 1 and len(backtick_segments) % 2 == 0

    def serialize_content_blocks(content_blocks, raw=False):
        content = ""
        for block in content_blocks:
            if block["type"] == "text":
                content = f"{content}{block['content'].strip()}\n"

            elif block["type"] == "reasoning":
                reasoning_display_content = "\n".join(
                    (f"> {line}" if not line.startswith(">") else line) for line in block["content"].splitlines()
                )
                reasoning_duration = block.get("duration", None)
                if reasoning_duration is not None:
                    if raw:
                        content = f'{content}\n<{block["tag"]}>{block["content"]}</{block["tag"]}>\n'
                    else:
                        content = f'{content}\n<details type="reasoning" done="true" duration="{reasoning_duration}">\n<summary>Thought for {reasoning_duration} seconds</summary>\n{reasoning_display_content}\n</details>\n'
                else:
                    if raw:
                        content = f'{content}\n<{block["tag"]}>{block["content"]}</{block["tag"]}>\n'
                    else:
                        content = f'{content}\n<details type="reasoning" done="false">\n<summary>Thinking…</summary>\n{reasoning_display_content}\n</details>\n'

            elif block["type"] == "code_interpreter":
                attributes = block.get("attributes", {})
                output = block.get("output", None)
                lang = attributes.get("lang", "")

                content_stripped, original_whitespace = (split_content_and_whitespace(content))
                if is_opening_code_block(content_stripped):
                    # Remove trailing backticks that would open a new block
                    content = content_stripped.rstrip("`").rstrip() + original_whitespace
                else:
                    # Keep content as is - either closing backticks or no backticks
                    content = content_stripped + original_whitespace

                if output:
                    output = html.escape(json.dumps(output))
                    if raw:
                        content = f'{content}\n<code_interpreter type="code" lang="{lang}">\n{block["content"]}\n</code_interpreter>\n```output\n{output}\n```\n'
                    else:
                        content = f'{content}\n<details type="code_interpreter" done="true" output="{output}">\n<summary>Analyzed</summary>\n```{lang}\n{block["content"]}\n```\n</details>\n'
                else:
                    if raw:
                        content = f'{content}\n<code_interpreter type="code" lang="{lang}">\n{block["content"]}\n</code_interpreter>\n'
                    else:
                        content = f'{content}\n<details type="code_interpreter" done="false">\n<summary>Analyzing...</summary>\n```{lang}\n{block["content"]}\n```\n</details>\n'
            else:
                block_content = str(block["content"]).strip()
                content = f"{content}{block['type']}: {block_content}\n"

        return content.strip()

    def tag_content_handler(content_type, tags, content, content_blocks):
        end_flag = False

        def extract_attributes(tag_content):
            """Extract attributes from a tag if they exist."""
            attributes = {}
            if not tag_content:  # Ensure tag_content is not None
                return attributes
            # Match attributes in the format: key="value" (ignores single quotes for simplicity)
            matches = re.findall(r'(\w+)\s*=\s*"([^"]+)"', tag_content)
            for key, value in matches:
                attributes[key] = value
            return attributes

        if content_blocks[-1]["type"] == "text":
            for tag in tags:
                # Match start tag e.g., <tag> or <tag attr="value">
                start_tag_pattern = rf"<{tag}(\s.*?)?>"
                match = re.search(start_tag_pattern, content)
                if not match:
                    continue

                attr_content = match.group(1) if match.group(1) else ""  # Ensure it's not None
                attributes = extract_attributes(attr_content)  # Extract attributes safely

                # Capture everything before and after the matched tag
                before_tag = content[: match.start()]  # Content before opening tag
                after_tag = content[match.end():]  # Content after opening tag

                # Remove the start tag and after from the currently handling text block
                content_blocks[-1]["content"] = content_blocks[-1]["content"].replace(match.group(0) + after_tag, "")

                if before_tag:
                    content_blocks[-1]["content"] = before_tag

                if not content_blocks[-1]["content"]:
                    content_blocks.pop()

                # Append the new block
                content_blocks.append(
                    {
                        "type": content_type,
                        "tag": tag,
                        "attributes": attributes,
                        "content": "",
                        "started_at": time.time(),
                    }
                )
                if after_tag:
                    content_blocks[-1]["content"] = after_tag

        elif content_blocks[-1]["type"] == content_type:
            tag = content_blocks[-1]["tag"]

            # Match end tag e.g., </tag>
            end_tag_pattern = rf"</{tag}>"

            # Check if the content has the end tag
            if re.search(end_tag_pattern, content):
                end_flag = True
                block_content = content_blocks[-1]["content"]

                # Strip start and end tags from the content
                start_tag_pattern = rf"<{tag}(.*?)>"
                block_content = re.sub(start_tag_pattern, "", block_content).strip()
                end_tag_regex = re.compile(end_tag_pattern, re.DOTALL)
                split_content = end_tag_regex.split(block_content, maxsplit=1)

                # Content inside the tag
                block_content = split_content[0].strip() if split_content else ""

                # Leftover content (everything after `</tag>`)
                leftover_content = split_content[1].strip() if len(split_content) > 1 else ""

                if block_content:
                    content_blocks[-1]["content"] = block_content
                    content_blocks[-1]["ended_at"] = time.time()
                    content_blocks[-1]["duration"] = int(
                        content_blocks[-1]["ended_at"] - content_blocks[-1]["started_at"]
                    )

                    # Reset the content_blocks by appending a new text block
                    if content_type != "code_interpreter":
                        if leftover_content:
                            content_blocks.append({"type": "text", "content": leftover_content})
                        else:
                            content_blocks.append({"type": "text", "content": ""})
                else:
                    # Remove the block if content is empty
                    content_blocks.pop()
                    if leftover_content:
                        content_blocks.append({"type": "text", "content": leftover_content})
                    else:
                        content_blocks.append({"type": "text", "content": ""})

                # Clean processed content
                content = re.sub(
                    rf"<{tag}(.*?)>(.|\n)*?</{tag}>",
                    "",
                    content,
                    flags=re.DOTALL,
                )

        return content, content_blocks, end_flag

    async def stream_wrapper(original_generator, _events):
        def wrap_item(item):
            return f"data: {item}\n\n"

        for event in _events:
            yield wrap_item(json.dumps(event))

        async for data in original_generator:
            yield data

    async def post_response_handler(response, events):
        async def stream_body_handler(response):
            nonlocal content
            nonlocal content_blocks

            async for line in response.body_iterator:
                line = line.decode("utf-8") if isinstance(line, bytes) else line

                # Skip empty lines and events that are not formatted
                if not line.strip() or not line.startswith("data:"):
                    continue
                _data = line

                # Remove the prefix
                _data = _data[len("data:"):].strip()

                try:
                    _data = json.loads(_data)
                    if "selected_model_id" in _data:
                        Chats.upsert_message_to_chat_by_id_and_message_id(
                            metadata["chat_id"],
                            metadata["message_id"],
                            {"selectedModelId": _data["selected_model_id"]},
                        )
                    else:
                        choices = _data.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        value = delta.get("content")
                        if not value:
                            continue

                        content = f"{content}{value}"
                        if not content_blocks:
                            content_blocks.append({"type": "text", "content": ""})
                        content_blocks[-1]["content"] = content_blocks[-1]["content"] + value
                        if DETECT_REASONING:
                            content, content_blocks, _ = (
                                tag_content_handler(
                                    "reasoning",
                                    REASONING_TAGS,
                                    content,
                                    content_blocks,
                                )
                            )

                        if DETECT_CODE_INTERPRETER:
                            content, content_blocks, end = (
                                tag_content_handler(
                                    "code_interpreter",
                                    CODE_INTERPRETER_TAGS,
                                    content,
                                    content_blocks,
                                )
                            )
                            if end:
                                break

                        if ENABLE_REALTIME_CHAT_SAVE:
                            # Save message in the database
                            Chats.upsert_message_to_chat_by_id_and_message_id(
                                metadata["chat_id"],
                                metadata["message_id"],
                                {"content": serialize_content_blocks(content_blocks)},
                            )
                        else:
                            _data = {"content": serialize_content_blocks(content_blocks)}
                    await event_emitter({"type": "chat:completion", "data": _data})
                except Exception as ex:
                    if "data: [DONE]" not in line:
                        log.debug("Error: ", ex)
                        continue

            # Clean up the last text block
            if content_blocks and content_blocks[-1]["type"] == "text":
                content_blocks[-1]["content"] = content_blocks[-1]["content"].strip()
                if not content_blocks[-1]["content"]:
                    content_blocks.pop()
                    if not content_blocks:
                        content_blocks.append({"type": "text", "content": ""})

            if response.background:
                await response.background()

        message = Chats.get_message_by_id_and_message_id(metadata["chat_id"], metadata["message_id"])
        last_assistant_message = get_last_assistant_message(form_data["messages"])
        content = (
            message.get("content", "") if message else last_assistant_message if last_assistant_message else ""
        )
        content_blocks = [{"type": "text", "content": content}]
        DETECT_REASONING = True
        DETECT_CODE_INTERPRETER = metadata.get("features", {}).get("code_interpreter", False)
        CODE_INTERPRETER_TAGS = ["code_interpreter"]
        REASONING_TAGS = [
            "think",
            "thinking",
            "reason",
            "reasoning",
            "thought",
            "Thought",
        ]
        try:
            for event in events:
                await event_emitter({"type": "chat:completion", "data": event})

                # Save message in the database
                Chats.upsert_message_to_chat_by_id_and_message_id(
                    metadata["chat_id"], metadata["message_id"], {**event}
                )

            # Start streaming the response content
            await stream_body_handler(response)

            if DETECT_CODE_INTERPRETER:
                MAX_RETRIES = 5
                retries = 0
                while content_blocks[-1]["type"] == "code_interpreter" and retries < MAX_RETRIES:
                    await event_emitter(
                        {
                            "type": "chat:completion",
                            "data": {
                                "content": serialize_content_blocks(content_blocks),
                            },
                        }
                    )
                    retries += 1
                    log.debug(f"Attempt count: {retries}")
                    output = ""
                    try:
                        if content_blocks[-1]["attributes"].get("type") == "code":
                            code = content_blocks[-1]["content"]
                            if CODE_INTERPRETER_ENGINE == "pyodide":
                                output = await event_caller(
                                    {
                                        "type": "execute:python",
                                        "data": {
                                            "id": str(uuid4()),
                                            "code": code,
                                            "session_id": metadata.get("session_id", None),
                                        },
                                    }
                                )
                            elif CODE_INTERPRETER_ENGINE == "jupyter":
                                output = await execute_code_jupyter(
                                    request.app.state.config.CODE_INTERPRETER_JUPYTER_URL,
                                    code,
                                    (
                                        request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_TOKEN
                                        if request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH == "token"
                                        else None
                                    ),
                                    (
                                        request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_PASSWORD
                                        if request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH == "password"
                                        else None
                                    ),
                                )
                            else:
                                output = {"stdout": "Code interpreter engine not configured."}

                            if isinstance(output, dict):
                                stdout = output.get("stdout", "")
                                if isinstance(stdout, str):
                                    stdoutLines = stdout.split("\n")
                                    output["stdout"] = "\n".join(stdoutLines)
                                result = output.get("result", "")
                                if isinstance(result, str):
                                    resultLines = result.split("\n")
                                    output["result"] = "\n".join(resultLines)
                    except Exception as e:
                        output = str(e)

                    content_blocks[-1]["output"] = output
                    content_blocks.append(
                        {
                            "type": "text",
                            "content": "",
                        }
                    )
                    await event_emitter(
                        {
                            "type": "chat:completion",
                            "data": {
                                "content": serialize_content_blocks(content_blocks),
                            },
                        }
                    )
                    try:
                        res = await generate_chat_completion(
                            request,
                            {
                                "model": model_id,
                                "stream": True,
                                "messages": [
                                    *form_data["messages"],
                                    {
                                        "role": "assistant",
                                        "content": serialize_content_blocks(
                                            content_blocks, raw=True
                                        ),
                                    },
                                ],
                            },
                            user,
                        )
                        if not isinstance(res, StreamingResponse):
                            break
                        await stream_body_handler(res)
                    except Exception as e:
                        log.debug(e)
                        break

            title = Chats.get_chat_title_by_id(metadata["chat_id"])
            data = {"done": True, "content": serialize_content_blocks(content_blocks), "title": title}
            if not ENABLE_REALTIME_CHAT_SAVE:
                # Save message in the database
                Chats.upsert_message_to_chat_by_id_and_message_id(
                    metadata["chat_id"], metadata["message_id"], {"content": serialize_content_blocks(content_blocks)},
                )

            # Send a webhook notification if the user is not active
            if get_active_status_by_user_id(user.id) is None:
                webhook_url = Users.get_user_webhook_url_by_id(user.id)
                if webhook_url:
                    post_webhook(
                        request.app.state.WEBUI_NAME,
                        webhook_url,
                        f"{title} - {request.app.state.config.WEBUI_URL}/c/{metadata['chat_id']}\n\n{content}",
                        {
                            "action": "chat",
                            "message": content,
                            "title": title,
                            "url": f"{request.app.state.config.WEBUI_URL}/c/{metadata['chat_id']}",
                        },
                    )
            await event_emitter({"type": "chat:completion", "data": data})
            await background_tasks_handler()

        except asyncio.CancelledError:
            await event_emitter({"type": "task-cancelled"})
            if not ENABLE_REALTIME_CHAT_SAVE:
                # Save message in the database
                Chats.upsert_message_to_chat_by_id_and_message_id(
                    metadata["chat_id"], metadata["message_id"], {"content": serialize_content_blocks(content_blocks)},
                )
        if response.background is not None:
            await response.background()

    event_caller = None
    if (
            "session_id" in metadata
            and metadata["session_id"]
            and "chat_id" in metadata
            and metadata["chat_id"]
            and "message_id" in metadata
            and metadata["message_id"]
    ):
        if event_caller is None:
            event_emitter = get_event_emitter(metadata)
        event_caller = get_event_call(metadata)

    # Streaming response
    if event_emitter and event_caller:
        model_id = form_data.get("model", "")
        Chats.upsert_message_to_chat_by_id_and_message_id(
            metadata["chat_id"],
            metadata["message_id"],
            {
                "model": model_id,
            },
        )
        task_id, _ = create_task(post_response_handler(response, events))
        return {"status": True, "task_id": task_id}
    else:
        return StreamingResponse(
            stream_wrapper(response.body_iterator, events),
            headers=dict(response.headers),
            background=response.background,
        )

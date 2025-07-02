# Chat UI routers
from src.controller.chat_ui.audio_controller import router as audio_router
from src.controller.chat_ui.auth_controller import router as auth_router
from src.controller.chat_ui.channel_controller import router as channel_router
from src.controller.chat_ui.chat_controller import router as chat_router
from src.controller.chat_ui.config_controller import router as config_router
from src.controller.chat_ui.feedback_controller import router as feedback_router
from src.controller.chat_ui.files_controller import router as files_router
from src.controller.chat_ui.folder_controller import router as folder_router
from src.controller.chat_ui.group_controller import router as group_router
from src.controller.chat_ui.knowledge_controller import router as knowledge_router
from src.controller.chat_ui.memory_controller import router as memory_router
from src.controller.chat_ui.model_controller import router as model_router
from src.controller.chat_ui.prompts_controller import router as prompts_router
from src.controller.chat_ui.task_controller import router as task_router
from src.controller.chat_ui.tool_controller import router as tool_router
from src.controller.chat_ui.user_controller import router as user_router
from src.controller.chat_ui.utils_controller import router as utils_router
from src.controller.chat_ui.weights_and_bias_controller import router as wb_router

# Misc routers
from src.controller.health_controller import router as health_router
from src.controller.rag_controller import router as rag_router

# List of all routers to be included in the FastAPI app
routers = [
    health_router,
    audio_router,
    auth_router,
    channel_router,
    chat_router,
    config_router,
    feedback_router,
    files_router,
    folder_router,
    group_router,
    knowledge_router,
    memory_router,
    model_router,
    prompts_router,
    task_router,
    tool_router,
    user_router,
    utils_router,
    rag_router,
    wb_router
]

import logging

from fastapi import APIRouter, status, HTTPException
from fastapi.responses import StreamingResponse

from src.model.request_validation import InferenceRequest, FineTuningRequest
from src.service.fine_tuning.data_preprocessor import data_preprocessor
from src.service.fine_tuning.lora_hyperparameters import lora_hyperparameters
from src.service.fine_tuning.model_service import model_service
from src.service.storage_manager import storage_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/v1', tags=['Inference Controller'])


@router.post("/completions", status_code=status.HTTP_200_OK)
async def model_inference(request: InferenceRequest):
    """
    Perform model inference with optional fine-tuned model.

    Args:
        request (InferenceRequest): Inference request with user id and history

    Returns:
        StreamingResponse of generated text
    """
    try:
        history_list = request.history
        user_id = request.user_id

        # Convert to a list of messages from the object
        history = [message.to_dict() for message in history_list]

        # Get the latest user message
        query = history.pop().get("content").strip()

        # Get the index file from the DB
        logger.info(f"Started fetching the existing index")

        # Check if an index is available for the user
        index_exist = await storage_manager.check_file_exists(
            email=user_id,
            file_name='index'
        )

        context = ''
        if index_exist:
            rag_service = await storage_manager.read(
                email=user_id,
                filename='index'
            )

            # Start search
            logger.info(f"Started querying the vector store")
            context_list = await rag_service.search(query=query)

            # Convert the list of context to string
            context = '\n'.join([f"{i + 1}: {line}" for i, line in enumerate(context_list)]) if context_list else ''

        return StreamingResponse(
            model_service.start_inference(
                user_id=user_id,
                user_query=query,
                history=history,
                context=context
            ),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fine-tune")
async def fine_tune_model(user_id: str, request: FineTuningRequest):
    """
    Fine-tune the language model with a custom dataset.

    This endpoint provides a step-by-step fine-tuning process:
    1. Convert dataset to text
    2. Preprocess the text data
    3. Calculate optimal LoRA hyperparameters
    4. Fine-tune the model
    5. Stream progress updates

    Args:
        user_id (str): Unique ID of the user
        request (FineTuningRequest): Fine-tuning configuration

    Returns:
        StreamingResponse of fine-tuning progress
    """
    try:
        # Convert dataset to text
        text_data = data_preprocessor.bytes_to_text(request.dataset)

        # Preprocess dataset
        processed_data = data_preprocessor.preprocess_text(text_data)

        # Calculate optimal hyperparameters
        dataset_size = len(processed_data)
        model_size = 1_000_000_000  # 1B for Llama 3.2
        hyperparams = lora_hyperparameters.calculate(
            model_size=model_size,
            dataset_size=dataset_size
        )

        # Start Fine-tuning
        async def fine_tune_progress():
            async for progress in model_service.fine_tune(
                    user_id=user_id,
                    dataset=processed_data,
                    r=hyperparams['r'],
                    lora_alpha=hyperparams['lora_alpha'],
                    num_epochs=request.num_epochs,
                    learning_rate=request.learning_rate
            ):
                yield f"progress: {progress}\n\n"

        return StreamingResponse(fine_tune_progress(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

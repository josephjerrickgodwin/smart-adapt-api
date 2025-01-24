import logging

from fastapi import APIRouter, status, HTTPException
from fastapi.responses import StreamingResponse

from src.model.request_validation import InferenceRequest, FineTuningRequest
from src.service.fine_tuning.data_preprocessor import data_preprocessor
from src.service.fine_tuning.lora_hyperparameters import lora_hyperparameters
from src.service.fine_tuning.model_finetuner import ModelFineTuner
from src.service.fine_tuning.model_service import model_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/v1', tags=['Inference Controller'])


@router.post("/completions", status_code=status.HTTP_200_OK)
async def model_inference(email: str, request: InferenceRequest):
    """
    Perform model inference with optional fine-tuned model.

    Args:
        email (str): Email address of the user
        request (InferenceRequest): Inference request with query and parameters

    Returns:
        StreamingResponse of generated text
    """
    try:
        # Use the base model for inference by default
        pipe = model_service.predict(
            email=email,
            use_lora=request.use_lora
        )

        # Generate text with streaming
        async def generate():
            # Use the pipeline with user-specified parameters
            outputs = pipe(
                request.query,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )

            # Stream generated text chunks
            for chunk in outputs[0]['generated_text'].split():
                yield f"data: {chunk}\n\n"
            yield f"[END]"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fine-tune")
async def fine_tune_model(email: str, request: FineTuningRequest):
    """
    Fine-tune the language model with a custom dataset.

    This endpoint provides a step-by-step fine-tuning process:
    1. Convert dataset to text
    2. Preprocess the text data
    3. Calculate optimal LoRA hyperparameters
    4. Fine-tune the model
    5. Stream progress updates

    Args:
        email (str): Email address of the
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

        # Fine-tuning
        async def fine_tune_progress():
            fine_tuner = ModelFineTuner()
            async for progress in fine_tuner.fine_tune(
                    dataset=processed_data,
                    model_name=request.model_name,
                    username=email,
                    r=hyperparams['r'],
                    lora_alpha=hyperparams['lora_alpha'],
                    num_epochs=request.num_epochs,
                    learning_rate=request.learning_rate
            ):
                yield f"progress: {progress}\n\n"

        return StreamingResponse(fine_tune_progress(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import logging
from datetime import datetime

from fastapi import APIRouter, status, HTTPException
from fastapi.responses import StreamingResponse

from src.model.request_validation import (
    DatasetInsertRequest,
    LoRAHyperparametersRequest
)
from src.service.fine_tuning.data_preprocessor import data_preprocessor
from src.service.fine_tuning.lora_hyperparameters import lora_hyperparameters

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/dataset', tags=['Data Controller'])


@router.post("/prune", status_code=status.HTTP_200_OK)
async def pre_process_dataset(email: str, request: DatasetInsertRequest):
    """
    Preprocess dataset for fine-tuning by removing unnecessary characters.

    Args:
        email (str): Email address of the user
        request (DatasetInsertRequest): Dataset insertion request

    Returns:
        StreamingResponse of preprocessed dataset
    """
    try:
        # Convert bytes to text
        logger.info(f"Create pre-process dataset request - Email: {email}, Time: {datetime.now()}")
        text_data = data_preprocessor.bytes_to_text(request.dataset)

        # Preprocess dataset
        processed_data = data_preprocessor.preprocess_text(text_data)

        # Stream preprocessed dataset
        async def generate():
            for chunk in processed_data:
                yield f"data: {chunk}\n\n"
            yield f"[END]"

        logger.info(f"Pre-process dataset request successful - Email: {email}, Time: {datetime.now()}")
        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Unexpected error in pre-processing the dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_200_OK,
            detail=str(e)
        )


@router.post("/hyperparameters", status_code=status.HTTP_200_OK)
async def calculate_lora_hyperparameters(email: str, request: LoRAHyperparametersRequest):
    """
    Calculate LoRA hyperparameters.

    Args:
        email (str): Email address of the user
        request (LoRAHyperparametersRequest): Hyperparameters request

    Returns:
        Dict of calculated LoRA hyperparameters
    """
    try:
        logger.info(f"Create lora hyperparameters request - Email: {email}, Time: {datetime.now()}")
        hyperparams = lora_hyperparameters.calculate(
            model_size=request.model_size,
            dataset_size=request.dataset_size
        )
        logger.info(f"successfully generated lora hyperparameters request: {email}, Time: {datetime.now()}")
        return hyperparams

    except ValueError as e:
        logger.warning(f"Lora hyperparameter attribute error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Invalid parameters detected: {str(e)}'
        )
    except Exception as e:
        logger.error(f"An internal server error occurred during generating lora hyperparameters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

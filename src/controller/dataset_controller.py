import json
import logging
from datetime import datetime

from fastapi import APIRouter, status, HTTPException
from fastapi.responses import Response

from src.exception.unicode_decode_error import UnicodeDecodeErrors
from src.model.request_validation import (
    DatasetInsertRequest,
    LoRAHyperparametersRequest
)
from src.service.fine_tuning.data_preprocessor import data_preprocessor
from src.service.fine_tuning.lora_hyperparameters import lora_hyperparameters

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/dataset', tags=['Data Controller'])


@router.post("/prune", status_code=status.HTTP_200_OK)
async def pre_process_dataset(user_id: str, request: DatasetInsertRequest):
    """
    Preprocess dataset for fine-tuning by removing unnecessary characters.

    Args:
        user_id (str): Unique ID of the user
        request (DatasetInsertRequest): Dataset insertion request

    Returns:
        StreamingResponse of preprocessed dataset
    """
    try:
        # Convert bytes to text
        logger.info(f"Create pre-process dataset request - user_id: {user_id}, Time: {datetime.now()}")
        text_data = data_preprocessor.bytes_to_text(request.dataset)

        # Preprocess dataset
        processed_data = data_preprocessor.preprocess_text(text_data)

        # Convert list to JSON string
        json_data = json.dumps(processed_data)

        logger.info(f"Pre-process dataset request successful - user_id: {user_id}, Time: {datetime.now()}")

        # Create a response with JSON content and headers for downloading
        return Response(
            content=json_data,
            media_type="application/json",
            headers={
                "Content-Disposition": "attachment; filename=data.json"
            }
        )

    except UnicodeDecodeErrors as e:
        logger.error(f"Pre-process dataset failed due to: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f'Invalid parameters detected: {str(e)}'
        )
    except Exception as e:
        logger.error(f"Unexpected error in pre-processing the dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/hyperparameters", status_code=status.HTTP_200_OK)
async def calculate_lora_hyperparameters(user_id: str, request: LoRAHyperparametersRequest):
    """
    Calculate LoRA hyperparameters.

    Args:
        user_id (str): Unique ID of the user
        request (LoRAHyperparametersRequest): Hyperparameters request

    Returns:
        Dict of calculated LoRA hyperparameters
    """
    try:
        logger.info(f"Create lora hyperparameters request - user_id: {user_id}, Time: {datetime.now()}")
        hyperparams = lora_hyperparameters.calculate(
            model_size=request.model_size,
            dataset_size=request.dataset_size
        )
        logger.info(f"successfully generated lora hyperparameters request: {user_id}, Time: {datetime.now()}")

        return Response(
            content=hyperparams,
            status_code=status.HTTP_200_OK,
        )

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

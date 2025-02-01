import logging
import pickle
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, status, HTTPException
from fastapi.responses import JSONResponse

from src.exception import DuplicateUserError
from src.exception.unicode_decode_error import UnicodeDecodeErrors
from src.model.index_model import IndexModel
from src.model.status_enum import Status
from src.service.fine_tuning.data_preprocessor import data_preprocessor
from src.service.rag_service import RAGService
from src.service.storage_manager import storage_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/index', tags=['Index Controller'])

INDEX_PREFIX = 'Index'


@router.post("/create", status_code=status.HTTP_201_CREATED)
async def create_index(data: IndexModel) -> JSONResponse:
    user_id = data.user_id
    user_data = data.data

    try:
        logger.info(f"Create index request - User: {user_id}, Time: {datetime.now()}")

        # Convert dataset to text
        text_data = data_preprocessor.bytes_to_text(user_data)

        # Preprocess dataset
        processed_data = data_preprocessor.preprocess_text(text_data)

        # Check for existing index
        try:
            logger.info(f"Started loading the index data")
            index_data = await storage_manager.read(
                user_id=user_id,
                filename=INDEX_PREFIX
            )

            # Initialize the index service
            index_data = index_data["data"]
            rag_service: RAGService = pickle.loads(index_data)

            # Generate embeddings for the new data
            logger.info("Generating embeddings for the data")
            embeddings = await rag_service.get_embeddings(processed_data)

            # Update the index module
            await rag_service.index_store.add_index(
                embeddings=embeddings,
                labels=processed_data
            )

            # Update the index
            logger.info(f"Updating the database")
            _ = await storage_manager.update(
                user_id=user_id,
                filename=INDEX_PREFIX,
                data=rag_service
            )

        except FileNotFoundError:
            logger.info(f"Started creating the index data")

            # Create the RAG service and configure the vector store (HNSW)
            rag_service = RAGService()

            # Generate embeddings for the new data
            logger.info("Generating embeddings for the data")
            embeddings = await rag_service.get_embeddings(processed_data)

            # Start the simulation
            logger.info('Started generating hyperparameters and simulation. This may take a while.')
            _ = await rag_service.configure_vector_store(
                embeddings=embeddings,
                docs=processed_data
            )

            # Upload the index to MongoDB
            _ = await storage_manager.create(
                user_id=user_id,
                filename=INDEX_PREFIX,
                data=rag_service
            )
            logger.info(f"Successfully created pickle file for user_id: {user_id}")

        # Get the optimal parameters
        optimal_params = await rag_service.get_optimal_hyperparameters()

        payload = {
            "user_id": user_id,
            "hyperparameters": optimal_params,
            "status": 'SUCCESS'
        }
        return JSONResponse(payload, status_code=status.HTTP_201_CREATED)

    except UnicodeDecodeErrors as e:
        logger.error(f"Pre-process dataset failed due to: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Invalid parameters detected: {str(e)}'
        )

    except DuplicateUserError:
        logger.warning(f"Attempted to create duplicate entry for user_id: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File already exists"
        )
    except Exception as e:
        logger.error(f"Unexpected error in creating an index: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


@router.get("/get/{user_id}", status_code=status.HTTP_200_OK)
async def get_index(user_id: str) -> JSONResponse:
    try:
        logger.info(f"Started fetching index information - User: {user_id}")

        # Get the index file from the DB
        index_data = await storage_manager.read(
            user_id=user_id,
            filename=INDEX_PREFIX
        )

        # Initialize the index service
        logger.info(f"Started loading the index data")
        index_data = index_data["data"]
        index_service: RAGService = pickle.loads(index_data)

        # Get relevant information
        hyperparameters = await index_service.get_optimal_hyperparameters()

        logger.info(f"Successfully retrieved the index information for user_id: {user_id}")
        return JSONResponse(hyperparameters)

    except FileNotFoundError:
        logger.warning(f"Attempted to retrieve non-existent index file for user_id: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Index does not exist!"
        )
    except Exception as e:
        logger.error(f"Unexpected error in getting index information for user_id: {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred! Please try again later!"
        )


@router.put("/update/{user_id}", status_code=status.HTTP_200_OK)
async def update_index(data: IndexModel) -> JSONResponse:
    user_id = data.user_id
    user_data = data.data

    try:
        logger.info(f"Started index update request - User: {user_id}")

        # Get the index file from the DB
        index_data = await storage_manager.read(
            user_id=user_id,
            filename=INDEX_PREFIX
        )

        # Initialize the index service
        logger.info(f"Started loading the index data")
        index_data = index_data["data"]
        index_service: RAGService = pickle.loads(index_data)

        # Convert dataset to text
        text_data = data_preprocessor.bytes_to_text(data.data)

        # Preprocess dataset
        processed_data = data_preprocessor.preprocess_text(text_data)

        # Generate embeddings for the new data
        logger.info(f"Generating embeddings for the data")
        embeddings = await index_service.get_embeddings(processed_data)

        # Update the index module
        await index_service.index_store.add_index(
            embeddings=embeddings,
            labels=user_data
        )

        # Update the index
        logger.info(f"Updating the database")
        _ = await storage_manager.update(
            user_id=user_id,
            filename=INDEX_PREFIX,
            data=index_service
        )

        payload = {
            "user_id": user_id,
            "index_count": len(index_service.index_store),
            "status": Status.SUCCESS.value
        }
        return JSONResponse(payload)

    except FileNotFoundError:
        logger.warning(f"Attempted to update non-existent file for user_id: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The requested file does not exist!"
        )
    except UnicodeDecodeErrors as e:
        logger.error(f"Pre-process dataset failed due to: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f'Invalid parameters detected: {str(e)}'
        )
    except Exception as e:
        logger.error(f"Unexpected error in update_pickle_file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later!"
        )


@router.delete("/delete/{user_id}", status_code=status.HTTP_200_OK)
async def delete_index(user_id: str) -> Dict[str, str]:
    try:
        logger.info(f"Started index delete request - user_id: {user_id}")

        # Get the index file from the DB
        await storage_manager.delete(user_id=user_id, filename=INDEX_PREFIX)

        logger.info(f"Successfully deleted index for user_id: {user_id}")

        return {"status": "Index has been deleted successfully"}

    except FileNotFoundError:
        logger.warning(f"Attempted to delete non-existent index for user_id: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The requested file does not exist!"
        )
    except Exception as e:
        logger.error(f"Unexpected error in delete_pickle_file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later!"
        )

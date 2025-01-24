import logging
import pickle
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, status, HTTPException
from fastapi.responses import JSONResponse

from src.exception import DuplicateEmailError
from src.model.index_model import IndexModel
from src.model.status_enum import Status
from src.service.rag_service import RAGService
from src.service.storage_manager import storage_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/index', tags=['Index Controller'])


@router.post("/create", status_code=status.HTTP_201_CREATED)
async def create_index(email: str, data: IndexModel) -> JSONResponse:
    try:
        logger.info(f"Create index request - Email: {email}, Time: {datetime.now()}")

        # Check for existing index
        try:
            _ = await storage_manager.read(
                email=email,
                filename='index'
            )
            # If there's an existing index, raise error
            raise DuplicateEmailError("")
        except FileNotFoundError:
            """ Ignore if the file is already exist """
            pass

        # Create the RAG service and configure the vector store (HNSW)
        rag_service = RAGService()
        _ = await rag_service.configure_vector_store(data.data)

        # Get the optimal parameters
        optimal_params = await rag_service.get_optimal_hyperparameters()

        # Upload the index to MongoDB
        file_name = await storage_manager.create(
            email=email,
            filename='index',
            data=rag_service
        )
        logger.info(f"Successfully created pickle file for email: {email}")

        payload = {
            "email": email,
            "hyperparameters": optimal_params,
            'file_name': file_name,
            "status": 'SUCCESS'
        }
        return JSONResponse(payload, status_code=status.HTTP_201_CREATED)

    except DuplicateEmailError:
        logger.warning(f"Attempted to create duplicate entry for email: {email}")
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


@router.get("/get/{email}", status_code=status.HTTP_200_OK)
async def get_index(email: str) -> JSONResponse:
    try:
        logger.info(f"Started fetching index information - Email: {email}")

        # Get the index file from the DB
        index_data = await storage_manager.read(
            email=email,
            filename='index'
        )

        # Initialize the index service
        logger.info(f"Started loading the index data")
        index_data = index_data["data"]
        index_service: RAGService = pickle.loads(index_data)

        # Get relevant information
        hyperparameters = await index_service.get_optimal_hyperparameters()

        logger.info(f"Successfully retrieved the index information for email: {email}")
        return JSONResponse(hyperparameters)

    except FileNotFoundError:
        logger.warning(f"Attempted to retrieve non-existent index file for email: {email}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Index does not exist!"
        )
    except Exception as e:
        logger.error(f"Unexpected error in getting index information for email: {email}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred! Please try again later!"
        )


@router.put("/update/{email}", status_code=status.HTTP_200_OK)
async def update_index(email: str, data: IndexModel) -> JSONResponse:
    try:
        logger.info(f"Started index update request - Email: {email}")

        # Get the index file from the DB
        index_data = await storage_manager.read(
            email=email,
            filename='index'
        )

        # Initialize the index service
        logger.info(f"Started loading the index data")
        index_data = index_data["data"]
        index_service: RAGService = pickle.loads(index_data)

        # Generate embeddings for the new data
        logger.info(f"Generating embeddings for the data")
        embeddings = await index_service.get_embeddings(data.data)

        # Update the index module
        await index_service.index_store.add_index(
            embeddings=embeddings,
            labels=data
        )

        # Update the index
        logger.info(f"Updating the database")
        _ = await storage_manager.update(
            email=email,
            filename='index',
            data=index_service
        )

        payload = {
            "email": email,
            "index_count": len(index_service.index_store),
            "status": Status.SUCCESS.value
        }
        return JSONResponse(payload)

    except FileNotFoundError:
        logger.warning(f"Attempted to update non-existent file for email: {email}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The requested file does not exist!"
        )
    except Exception as e:
        logger.error(f"Unexpected error in update_pickle_file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later!"
        )


@router.delete("/delete/{email}", status_code=status.HTTP_200_OK)
async def delete_index(email: str) -> Dict[str, str]:
    try:
        logger.info(f"Started index delete request - Email: {email}")

        # Get the index file from the DB
        await storage_manager.delete(email=email, filename='index')

        logger.info(f"Successfully deleted index for email: {email}")

        return {"status": "Index has been deleted successfully"}

    except FileNotFoundError:
        logger.warning(f"Attempted to delete non-existent index for email: {email}")
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

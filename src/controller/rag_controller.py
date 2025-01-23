import logging
import pickle

from fastapi import APIRouter, status, HTTPException
from fastapi.responses import JSONResponse

from src.service.rag_service import RAGService
from src.service.storage_manager import storage_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/rag', tags=['RAG Controller'])


@router.post("/search", status_code=status.HTTP_200_OK)
async def search(email: str, query: str, top_k: int = 0):
    try:
        logger.info(f"Started RAG search service for user: {email}")

        # Get the index file from the DB
        logger.info(f"Started fetching the existing index")
        index_data = await storage_manager.read(
            email=email,
            filename='index'
        )

        # Initialize the index service
        logger.info(f"Started loading the index data")
        index_data = index_data["data"]
        index_service: RAGService = pickle.loads(index_data)

        # Start search
        logger.info(f"Started querying the vector store")
        results = await index_service.search(query=query, k=top_k)

        return JSONResponse(results)

    except FileNotFoundError:
        logger.warning(f"Attempted to delete non-existent index for email: {email}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    except Exception as e:
        logger.error(f"Unexpected error in deleting index for email: {email} - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later!"
        )

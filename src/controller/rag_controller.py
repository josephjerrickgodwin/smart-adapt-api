import logging

from fastapi import APIRouter, status, HTTPException
from fastapi.responses import JSONResponse

from src.model.rag_model import RAGSearchModel
from src.service.storage_manager import storage_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/rag', tags=['RAG Controller'])


@router.post("/search", status_code=status.HTTP_200_OK)
async def search(user_id: str, data: RAGSearchModel):
    try:
        query = data.query
        top_k = data.top_k if data.top_k else 0

        logger.info(f"Started RAG search service for user: {user_id}")

        # Get the index file from the DB
        logger.info(f"Started fetching the existing index")
        rag_service = await storage_manager.read(
            user_id=user_id,
            filename='index'
        )

        # Start search
        logger.info(f"Started querying the vector store")
        results = await rag_service.search(query=query, k=top_k)

        return JSONResponse(results)

    except FileNotFoundError:
        logger.warning(f"An index does not exist for user_id: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Index does not exist for user_id!"
        )
    except Exception as e:
        logger.error(f"Unexpected error while querying the index for user_id: {user_id} - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later!"
        )

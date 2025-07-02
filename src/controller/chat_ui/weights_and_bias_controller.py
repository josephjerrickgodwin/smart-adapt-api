from aiohttp import ClientResponseError
from fastapi import APIRouter, Depends, Query, HTTPException, status

from src.service.utils.auth_service import get_verified_user
from src.service.weights_and_bias_service import wb_service

router = APIRouter(prefix="/api/v1/wb", tags=["Weights and Bias"])


@router.get("/status")
async def get_wb(
        knowledge_id: str = Query(..., description="Knowledge tag"),
        user_id: str = Query(..., description="User tag"),
        user=Depends(get_verified_user)
):
    try:
        return await wb_service.fetch_run_history(
            knowledge_id=knowledge_id,
            user_id=user_id
        )
    except ClientResponseError as e:
        raise HTTPException(
            status_code=e.status,
            detail=e.message
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

from fastapi import APIRouter, status

router = APIRouter(prefix='/health', tags=['Health Controller'])


@router.get("", status_code=status.HTTP_200_OK)
async def health() -> str:
    return "The service is up and running"

import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import uvicorn
from fastapi import FastAPI
from src.controller import (
    health_router,
    index_router,
    dataset_router,
    inference_router,
    rag_router
)

app = FastAPI()

# Add routes
app.include_router(health_router)
app.include_router(index_router)
app.include_router(dataset_router)
app.include_router(inference_router)
app.include_router(rag_router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="debug"
    )

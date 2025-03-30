import io
import logging
import os
import pickle
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import ftfy
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from src.model.constants import ERROR_MESSAGES
from src.model.files import (
    FileModel,
    FileModelResponse,
    Files, FileMeta,
)
from src.service.embedding_service import embedding_service
from src.service.env import SRC_LOG_LEVELS
from src.service.fine_tuning.data_preprocessor import data_preprocessor
from src.service.rag_service import RAGService
from src.service.utils.auth_service import get_admin_user, get_verified_user
from src.service.utils.storage.document_loader import Loader
from src.service.utils.storage.storage_service import Storage

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter(prefix="/api/v1/files", tags=["files"])


############################
# Upload File
############################

@router.post("/", response_model=FileModelResponse)
async def upload_file(
        request: Request,
        session_id: str,
        file: UploadFile = File(...),
        user=Depends(get_verified_user)
):
    log.info(f"file.content_type: {file.content_type}")
    try:
        created_at = int(time.time())
        un_sanitized_filename = file.filename
        filename = os.path.basename(un_sanitized_filename)

        # Replace filename with uuid
        id = str(uuid.uuid4())
        data_filename = f"{user.id}__index.pkl"
        temp_filename = f"tmp-{id}_{filename}"

        # Upload the fie (temporarily)
        _, file_path = Storage.upload_file(file.file, temp_filename)

        # Initialize the document loader
        loader = Loader(
            engine=request.app.state.config.CONTENT_EXTRACTION_ENGINE,
            TIKA_SERVER_URL=request.app.state.config.TIKA_SERVER_URL,
            PDF_EXTRACT_IMAGES=request.app.state.config.PDF_EXTRACT_IMAGES,
        )
        docs = loader.get_loader(
            filename=filename,
            file_content_type=file.content_type,
            file_path=file_path
        ).load()

        # Clear the temporary file
        Storage.delete_file(temp_filename)

        # Pre-process the data
        documents = []
        for doc in docs:
            # Fix any inconsistencies
            contents = ftfy.fix_text(doc.page_content)

            # Pre-process the data
            processed_data = data_preprocessor.preprocess_text(contents)

            # Extract the metadata
            metadata = doc.metadata

            # Extract the page number
            page_number = metadata.get('page', 0)

            # Add metadata to each chunk
            processed_chunks = [
                f'<header>\nfile name: {file.filename}\nPage: {page_number}</header>\n{chunk}'
                for chunk in processed_data
            ]

            # Update the documents for indexing
            documents.extend(processed_chunks)

        # Check for an existing index store for the current session
        rag_service = Storage.get_file(data_filename)
        if rag_service:
            # Generate embeddings for the new data
            log.info("Generating embeddings for the data")
            embeddings = await embedding_service.get_embeddings(documents)

            # Get the existing hyperparameters
            hyperparameters = rag_service.get_optimal_hyperparameters()

            # Initialize a new session (If not already exist)
            await rag_service.index_store.create_session_for_index(
                session_id=session_id,
                ef_construction=hyperparameters['ef_construction'],
                ef_search=hyperparameters['ef_search'],
                m=hyperparameters['m']
            )

            # Add new data to the index
            await rag_service.index_store.add_index(
                session_id=session_id,
                vectors=embeddings,
                labels=documents
            )

            # Remove the existing index store from the storage
            Storage.delete_file(data_filename)
        else:
            # Create the RAG service and configure the vector store (HNSW)
            rag_service = RAGService()

            # Generate embeddings for the new data
            log.info("Generating embeddings for the data")
            embeddings = await embedding_service.get_embeddings(documents)

            # Start the simulation
            log.info('Started generating hyperparameters and simulation. This may take a while.')
            _ = await rag_service.configure_vector_store(
                session_id=session_id,
                embeddings=embeddings,
                docs=documents
            )

        # Serialize with pickle
        pickle_bytes = pickle.dumps(rag_service)

        # Convert to BinaryIO
        binary_stream = io.BytesIO(pickle_bytes)
        binary_stream.seek(0)

        # Save the index store
        _, _ = Storage.upload_file(
            file=binary_stream,
            filename=data_filename
        )

        updated_at = int(time.time())
        meta = FileMeta(
            name=filename,
            content_type=file.content_type,
            size=file.size,
            model_config={}
        )
        file_item = FileModelResponse(
            id=id,
            user_id=user.id,
            filename=filename,
            meta=meta,
            created_at=created_at,
            updated_at=updated_at,
            model_config={}
        )
        return file_item

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT,
        )


############################
# List Files
############################


@router.get("/", response_model=list[FileModelResponse])
async def list_files(user=Depends(get_verified_user)):
    if user.role == "admin":
        files = Files.get_files()
    else:
        files = Files.get_files_by_user_id(user.id)
    return files


############################
# Delete All Files
############################


@router.delete("/all")
async def delete_all_files(user=Depends(get_admin_user)):
    result = Files.delete_all_files()
    if result:
        try:
            Storage.delete_all_files()
        except Exception as e:
            log.exception(e)
            log.error("Error deleting files")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error deleting files"),
            )
        return {"message": "All files deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Error deleting files"),
        )


############################
# Get File By Id
############################


@router.get("/{id}", response_model=Optional[FileModel])
async def get_file_by_id(id: str, user=Depends(get_verified_user)):
    file = Files.get_file_by_id(id)

    if file and (file.user_id == user.id or user.role == "admin"):
        return file
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# Get File Data Content By Id
############################


@router.get("/{id}/data/content")
async def get_file_data_content_by_id(id: str, user=Depends(get_verified_user)):
    file = Files.get_file_by_id(id)

    if file and (file.user_id == user.id or user.role == "admin"):
        return {"content": file.data.get("content", "")}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# Get File Content By Id
############################


@router.get("/{id}/content")
async def get_file_content_by_id(id: str, user=Depends(get_verified_user)):
    file = Files.get_file_by_id(id)
    if file and (file.user_id == user.id or user.role == "admin"):
        try:
            file_path = Storage.get_file(file.path)
            file_path = Path(file_path)

            # Check if the file already exists in the cache
            if file_path.is_file():
                # Handle Unicode filenames
                filename = file.meta.get("name", file.filename)
                encoded_filename = quote(filename)  # RFC5987 encoding

                headers = {}
                if file.meta.get("content_type") not in [
                    "application/pdf",
                    "text/plain",
                ]:
                    headers = {
                        **headers,
                        "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}",
                    }

                return FileResponse(file_path, headers=headers)

            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=ERROR_MESSAGES.NOT_FOUND,
                )
        except Exception as e:
            log.exception(e)
            log.error("Error getting file content")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error getting file content"),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


@router.delete("/{id}")
async def delete_file_by_id(id: str, user=Depends(get_verified_user)):
    file = Files.get_file_by_id(id)
    if file and (file.user_id == user.id or user.role == "admin"):
        # We should add Chroma cleanup here

        result = Files.delete_file_by_id(id)
        if result:
            try:
                Storage.delete_file(file.path)
            except Exception as e:
                log.exception(e)
                log.error("Error deleting files")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.DEFAULT("Error deleting files"),
                )
            return {"message": "File deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error deleting file"),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

import io
import logging
from io import BytesIO
from typing import Optional, Tuple, List, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from src.model.constants import ERROR_MESSAGES
from src.model.files import Files, FileModel
from src.model.knowledge import (
    Knowledges,
    KnowledgeForm,
    KnowledgeResponse,
    KnowledgeUserResponse, )
from src.service.client_service import client_service
from src.service.env import SRC_LOG_LEVELS
from src.service.utils.access_control_service import has_access, has_permission
from src.service.utils.auth_service import get_verified_user

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


def get_knowledge_files(knowledge_bases: list):
    knowledge_with_files = {}
    knowledge_ids = []
    for knowledge_base in knowledge_bases:
        files = []
        if knowledge_base.data:
            files = Files.get_file_metadatas_by_ids(
                knowledge_base.data.get("file_ids", [])
            )

            # Check if all files exist
            if len(files) != len(knowledge_base.data.get("file_ids", [])):
                missing_files = list(
                    set(knowledge_base.data.get("file_ids", []))
                    - set([file.id for file in files])
                )
                if missing_files:
                    data = knowledge_base.data or {}
                    file_ids = data.get("file_ids", [])

                    for missing_file in missing_files:
                        file_ids.remove(missing_file)

                    data["file_ids"] = file_ids
                    Knowledges.update_knowledge_data_by_id(
                        id=knowledge_base.id, data=data
                    )

                    files = Files.get_file_metadatas_by_ids(file_ids)

        knowledge_with_files[knowledge_base.id] = KnowledgeUserResponse(
            **knowledge_base.model_dump(),
            files=files,
        )
        knowledge_ids.append(knowledge_base.id)
    return knowledge_ids, knowledge_with_files


@router.get("/", response_model=list[KnowledgeUserResponse])
async def get_knowledge(user=Depends(get_verified_user)):
    if user.role == "admin":
        knowledge_bases = Knowledges.get_knowledge_bases()
    else:
        knowledge_bases = Knowledges.get_knowledge_bases_by_user_id(user.id, "read")

    # Get files for each knowledge base
    knowledge_ids, knowledge_with_files = get_knowledge_files(knowledge_bases)

    # Check the knowledge status with the client
    try:
        knowledge_files = await client_service.get_knowledge_data_using_client(
            user_role='admin' if user.role == "admin" else 'user',
            knowledge_ids=knowledge_ids
        )
        if knowledge_files is not None:
            for knowledge_file in knowledge_files:
                knowledge_id = knowledge_file.get('id', '')
                knowledge_data = knowledge_file.get('data', None)
                if not knowledge_data:
                    continue
                if knowledge_id in knowledge_with_files:
                    knowledge_with_files[knowledge_id].data = knowledge_data

    except Exception as e:
        log.exception(
            'Error validating user knowledge with the client. '
            'Falling back to default status. '
            f'Traceback: {str(e)}'
        )

    return knowledge_with_files.values()


@router.get("/list", response_model=list[KnowledgeUserResponse])
async def get_knowledge_list(user=Depends(get_verified_user)):
    if user.role == "admin":
        knowledge_bases = Knowledges.get_knowledge_bases()
    else:
        knowledge_bases = Knowledges.get_knowledge_bases_by_user_id(user.id, "write")

    # Get files for each knowledge base
    knowledge_ids, knowledge_with_files = get_knowledge_files(knowledge_bases)

    # Check the knowledge status with the client
    try:
        knowledge_files = await client_service.get_knowledge_data_using_client(
            user_role='admin' if user.role == "admin" else 'user',
            knowledge_ids=knowledge_ids
        )
        if knowledge_files is not None:
            for knowledge_file in knowledge_files:
                knowledge_id = knowledge_file.get('id', '')
                knowledge_data = knowledge_file.get('data', None)
                if not knowledge_data:
                    continue
                if knowledge_id in knowledge_with_files:
                    knowledge_with_files[knowledge_id].data = knowledge_data

    except Exception as e:
        log.exception(
            'Error validating user knowledge with the client. '
            'Falling back to default status. '
            f'Traceback: {str(e)}'
        )

    validated_knowledge = knowledge_with_files.values()
    return validated_knowledge


@router.post("/create", response_model=Optional[KnowledgeResponse])
async def create_new_knowledge(
        request: Request,
        name: str = Form(...),
        description: str = Form(...),
        question_column_name: str = Form(...),
        answer_column_name: str = Form(...),
        file: UploadFile = File(...),
        access_control: Optional[str] = Form(None),
        user=Depends(get_verified_user),
):
    if not has_permission(user.id, "workspace.knowledge", request.app.state.config.USER_PERMISSIONS):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    # Extract file name with extension
    filename = file.filename
    file_size = file.size

    # Read the file
    file_io = await file.read()
    file_stream = BytesIO(file_io)

    # Load the dataset
    if filename.endswith(".xlsx"):
        df = pd.read_excel(file_stream)
    elif filename.endswith(".csv"):
        df = pd.read_csv(file_stream)
    elif filename.endswith(".json"):
        df = pd.read_json(file_stream, lines=False)
    elif filename.endswith(".jsonl"):
        df = pd.read_json(file_stream, lines=True)
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported file type!"
        )

    # Check for required columns
    columns_exist = all(col in df.columns for col in [question_column_name, answer_column_name])
    if not columns_exist:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The specified column for either question or answer does not exist!"
        )

    # Check all the knowledge for existing data
    Knowledge_list = Knowledges.get_knowledge_bases()
    for kb in Knowledge_list:
        current_kb_data = kb.data
        if not current_kb_data:
            continue

        # Check for file name and size
        knowledge_filename = current_kb_data.get('filename', '')
        knowledge_size = current_kb_data.get('size', 0)
        knowledge_status = current_kb_data.get('status', '')

        if knowledge_filename == filename and knowledge_size == file_size:
            if knowledge_status == 'Failed':
                _ = Knowledges.delete_knowledge_by_id(kb.id)
            elif knowledge_status == 'In Progress':
                _ = Knowledges.delete_knowledge_by_id(kb.id)
                # raise HTTPException(
                #     status_code=status.HTTP_400_BAD_REQUEST,
                #     detail="The knowledge is currently being added to the model!"
                # )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The knowledge already exist!"
                )

    # Add file information
    file_data = {
        "filename": filename,
        "size": file_size,
        "status": "In Progress"
    }

    # Create the knowledge record
    knowledge = KnowledgeForm(
        name=name,
        description=description,
        data=file_data,
        access_control=access_control
    )
    knowledge = Knowledges.insert_new_knowledge(user.id, knowledge)

    # Make sure that the knowledge does not exist already
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.FILE_EXISTS,
        )

    # Create a BytesIO stream (binary stream)
    bytes_stream = io.BytesIO()

    # Write the DataFrame to CSV using the text wrapper
    df.to_csv(bytes_stream, index=False)

    # Seek to the beginning of the BytesIO stream to read its content
    bytes_stream.seek(0)

    try:
        # Start fine-tuning
        _ = await client_service.fine_tuning_using_client(
            user_id=user.id,
            knowledge_id=knowledge.id,
            question_column_name=question_column_name,
            answer_column_name=answer_column_name,
            file_stream=bytes_stream
        )
        file_data['status'] = 'Completed'
        _ = Knowledges.update_knowledge_data_by_id(
            id=knowledge.id,
            data=file_data
        )
    except Exception as ex:
        log.error(f'Fine-tuning error due to: {str(ex)}')
        file_data['status'] = 'Failed'
        _ = Knowledges.update_knowledge_data_by_id(
            id=knowledge.id,
            data=file_data
        )

    return knowledge


class KnowledgeFilesResponse(KnowledgeResponse):
    files: list[FileModel]


@router.get("/{id}", response_model=Optional[KnowledgeFilesResponse])
async def get_knowledge_by_id(id: str, user=Depends(get_verified_user)):
    knowledge = Knowledges.get_knowledge_by_id(id=id)

    if knowledge:
        if (
            knowledge.user_id == user.id or 
            has_access(user.id, "read", knowledge.access_control)
        ):

            file_ids = knowledge.data.get("file_ids", []) if knowledge.data else []
            files = Files.get_files_by_ids(file_ids)

            return KnowledgeFilesResponse(
                **knowledge.model_dump(),
                files=files,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


@router.delete("/{id}/delete", response_model=bool)
async def delete_knowledge_by_id(id: str, user=Depends(get_verified_user)):
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    log.info(f"Deleting knowledge base: {id} (name: {knowledge.name})")
    _ = await client_service.remove_lora_adapter_using_client(
        user_id=user.id,
        knowledge_id=id
    )

    # Remove the knowledge data
    result = Knowledges.delete_knowledge_by_id(id=id)
    return result


@router.get("/adapter/download")
async def download_lora_adapter(user_id: str, knowledge_id: str, user=Depends(get_verified_user)):
    try:
        file_stream = await client_service.download_lora_adapter_using_client(user_id, knowledge_id)
        if file_stream is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="LoRA adapter not found."
            )
        return StreamingResponse(
            file_stream,
            media_type="application/x-zip-compressed",
            headers={
                "Content-Disposition": f"attachment; filename=adapter.zip"
            }
        )
    except HTTPException as e:
        log.error(f"Failed to download LoRA adapter from the client: {str(e)}")
        raise HTTPException(
            status_code=e.status_code, 
            detail=e.detail
        )
    except Exception as e:
        log.error(f"Failed to download LoRA adapter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to download LoRA adapter."
        )


@router.post("/fine-tune/stop")
async def stop_knowledge_training(
    user_id: str = Form(...),
    knowledge_id: str = Form(...),
    user=Depends(get_verified_user)
):
    """Stop the fine-tuning process for a knowledge base."""
    # Check if knowledge exists and user has permission
    knowledge = Knowledges.get_knowledge_by_id(id=knowledge_id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (not has_access(user.id, "write", knowledge.access_control)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    try:
        result = await client_service.stop_fine_tuning_using_client(user_id, knowledge_id)
        return result
    except Exception as e:
        log.error(f"Failed to stop training process: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop training process."
        )

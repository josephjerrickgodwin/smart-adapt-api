import logging
import os
import shutil
from io import BytesIO
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File, Form, BackgroundTasks

from src.model.constants import ERROR_MESSAGES
from src.model.files import Files, FileModel
from src.model.knowledge import (
    Knowledges,
    KnowledgeForm,
    KnowledgeResponse,
    KnowledgeUserResponse, )
from src.service.env import SRC_LOG_LEVELS
from src.service.fine_tuning.model_service import model_service
from src.service.storage_manager import storage_manager
from src.service.utils.access_control_service import has_access, has_permission
from src.service.utils.auth_service import get_verified_user

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


def get_knowledge_files(knowledge_bases: list) -> list[KnowledgeUserResponse]:
    knowledge_with_files = []
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

        knowledge_with_files.append(
            KnowledgeUserResponse(
                **knowledge_base.model_dump(),
                files=files,
            )
        )
    return knowledge_with_files


@router.get("/", response_model=list[KnowledgeUserResponse])
async def get_knowledge(user=Depends(get_verified_user)):
    if user.role == "admin":
        knowledge_bases = Knowledges.get_knowledge_bases()
    else:
        knowledge_bases = Knowledges.get_knowledge_bases_by_user_id(user.id, "read")

    # Get files for each knowledge base
    knowledge_with_files = get_knowledge_files(knowledge_bases)

    return knowledge_with_files


@router.get("/list", response_model=list[KnowledgeUserResponse])
async def get_knowledge_list(user=Depends(get_verified_user)):
    if user.role == "admin":
        knowledge_bases = Knowledges.get_knowledge_bases()
    else:
        knowledge_bases = Knowledges.get_knowledge_bases_by_user_id(user.id, "write")

    # Get files for each knowledge base
    knowledge_with_files = get_knowledge_files(knowledge_bases)

    return knowledge_with_files


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
        background_tasks: BackgroundTasks = None
):
    if user.role != "admin" and not has_permission(
        user.id, "workspace.knowledge", request.app.state.config.USER_PERMISSIONS
    ):
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
                # Remove the knowledge
                _ = Knowledges.delete_knowledge_by_id(kb.id)
                break
            elif knowledge_status == 'In Progress':
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The knowledge is currently being added to the model!"
                )
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

    # # Add the background task
    # background_tasks.add_task(
    #     model_service.fine_tuning_handler(
    #         df=df,
    #         user_id=user.id,
    #         knowledge_id=knowledge.id,
    #         question_column_name=question_column_name,
    #         answer_column_name=answer_column_name,
    #         file_data=file_data
    #     )
    # )

    return knowledge


class KnowledgeFilesResponse(KnowledgeResponse):
    files: list[FileModel]


@router.get("/{id}", response_model=Optional[KnowledgeFilesResponse])
async def get_knowledge_by_id(id: str, user=Depends(get_verified_user)):
    knowledge = Knowledges.get_knowledge_by_id(id=id)

    if knowledge:
        if (
            user.role == "admin"
            or knowledge.user_id == user.id
            or has_access(user.id, "read", knowledge.access_control)
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

    if (
        not has_access(user.id, "write", knowledge.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    log.info(f"Deleting knowledge base: {id} (name: {knowledge.name})")

    # Check for model Knowledge
    userdata_dir = storage_manager.get_user_dir(user.id)
    lora_path = model_service.get_lora_path(
        data_path=userdata_dir,
        knowledge_id=id
    )
    logs_path = model_service.get_logs_path(
        data_path=userdata_dir,
        knowledge_id=id
    )

    # Remove knowledge, if found
    if os.path.exists(lora_path) and len(os.listdir(lora_path)):
        shutil.rmtree(lora_path)
    if os.path.exists(logs_path) and len(os.listdir(logs_path)):
        shutil.rmtree(logs_path)

    # Remove the knowledge data
    result = Knowledges.delete_knowledge_by_id(id=id)
    return result

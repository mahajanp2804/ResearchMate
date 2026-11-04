from typing import Annotated

from fastapi import APIRouter, File, Query, Response, UploadFile, status
from fastapi.responses import FileResponse

from app.dependencies import DbSession, SettingsDep
from app.exceptions import InvalidInputError
from app.schemas.common import ErrorDetail
from app.schemas.document import (
    DocumentDetail,
    DocumentListResponse,
    DocumentSummary,
    UploadResponse,
    UploadResult,
)
from app.services import document_service

router = APIRouter(prefix="/api/documents", tags=["documents"])

UPLOAD_PATH = "/api/documents/upload"


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={201: {"description": "At least one new document was stored"}},
)
def upload_documents(
    files: Annotated[list[UploadFile], File(description="One or more PDF files")],
    db: DbSession,
    settings: SettingsDep,
    response: Response,
) -> UploadResponse:
    """Upload PDFs. Each file gets its own result: created, duplicate or rejected.

    Returns 201 when at least one new document was stored, otherwise 200.
    """
    if len(files) > settings.max_files_per_upload:
        raise InvalidInputError(
            f"Upload at most {settings.max_files_per_upload} files at a time.",
            code="too_many_files",
        )

    results: list[UploadResult] = []
    for upload in files:
        outcome = document_service.ingest_upload(db, settings, upload.filename, upload.file)
        results.append(
            UploadResult(
                filename=outcome.filename,
                status=outcome.status,
                document=(
                    DocumentSummary.model_validate(outcome.document) if outcome.document else None
                ),
                error=(
                    ErrorDetail(
                        code=outcome.error.code,
                        message=outcome.error.message,
                        details=outcome.error.details,
                    )
                    if outcome.error
                    else None
                ),
            )
        )

    created = sum(result.status == "created" for result in results)
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return UploadResponse(
        results=results,
        created=created,
        duplicates=sum(result.status == "duplicate" for result in results),
        rejected=sum(result.status == "rejected" for result in results),
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    documents, total = document_service.list_documents(db, limit=limit, offset=offset)
    return DocumentListResponse(
        items=[DocumentSummary.model_validate(document) for document in documents], total=total
    )


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(document_id: str, db: DbSession) -> DocumentDetail:
    return DocumentDetail.model_validate(document_service.get_document(db, document_id))


@router.get(
    "/{document_id}/file",
    response_class=FileResponse,
    responses={200: {"content": {"application/pdf": {}}}},
)
def get_document_file(document_id: str, db: DbSession, settings: SettingsDep) -> FileResponse:
    """Serve the original PDF inline, so citations can open it at a page (``#page=N``)."""
    document, path = document_service.get_document_file(db, settings, document_id)
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=document.filename,
        content_disposition_type="inline",
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, db: DbSession, settings: SettingsDep) -> Response:
    document_service.delete_document(db, settings, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

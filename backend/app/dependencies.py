"""FastAPI dependencies shared by routers."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.config import Settings
from app.database import Database
from app.services.evaluation_service import EvaluationService
from app.services.ingestion_service import IngestionService
from app.services.paper_profile_service import PaperProfileService
from app.services.rag_pipeline import RagPipeline
from app.services.research_gap_service import ResearchGapService
from app.services.retrieval_service import RetrievalService


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_ingestion(request: Request) -> IngestionService:
    return request.app.state.ingestion


def get_retrieval(request: Request) -> RetrievalService:
    return request.app.state.retrieval


def get_rag(request: Request) -> RagPipeline:
    return request.app.state.rag


def get_profiles(request: Request) -> PaperProfileService:
    return request.app.state.profiles


def get_gaps(request: Request) -> ResearchGapService:
    return request.app.state.gaps


def get_evaluation(request: Request) -> EvaluationService:
    return request.app.state.evaluation


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_db(database: Annotated[Database, Depends(get_database)]) -> Iterator[Session]:
    session = database.session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
DatabaseDep = Annotated[Database, Depends(get_database)]
DbSession = Annotated[Session, Depends(get_db)]
IngestionDep = Annotated[IngestionService, Depends(get_ingestion)]
RetrievalDep = Annotated[RetrievalService, Depends(get_retrieval)]
RagDep = Annotated[RagPipeline, Depends(get_rag)]
ProfilesDep = Annotated[PaperProfileService, Depends(get_profiles)]
GapsDep = Annotated[ResearchGapService, Depends(get_gaps)]
EvaluationDep = Annotated[EvaluationService, Depends(get_evaluation)]

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.document import Document


class DocumentPage(Base):
    """Raw extracted text of one PDF page.

    Keeping page text means chunking settings can change and documents can be
    re-chunked without re-parsing the PDF, and citations can show the source page.
    """

    __tablename__ = "document_pages"
    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_page_document_number"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-based
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Non-whitespace characters; used to flag pages without extractable text.
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    has_text: Mapped[bool] = mapped_column(Boolean, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="pages")

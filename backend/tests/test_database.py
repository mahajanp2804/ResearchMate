"""Bringing a database made by an earlier version up to date."""

from pathlib import Path

import pytest
from sqlalchemy import inspect

from app.database import Database


@pytest.fixture
def database(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'old.db'}")
    database.create_tables()
    yield database
    database.dispose()


def columns(database: Database, table: str) -> set[str]:
    return {column["name"] for column in inspect(database.engine).get_columns(table)}


def test_new_nullable_columns_are_added_to_an_existing_table(database: Database) -> None:
    # A database from before the column existed.
    with database.engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE evaluation_questions DROP COLUMN category")
    assert "category" not in columns(database, "evaluation_questions")

    assert database.add_missing_columns() == ["evaluation_questions.category"]
    assert "category" in columns(database, "evaluation_questions")
    assert database.add_missing_columns() == []  # nothing left to do


def test_a_missing_required_column_is_refused(database: Database) -> None:
    with database.engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE evaluation_questions DROP COLUMN question")
    with pytest.raises(RuntimeError, match="evaluation_questions.question"):
        database.add_missing_columns()

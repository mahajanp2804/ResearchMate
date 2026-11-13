"""Database engine and session management.

A ``Database`` object owns one engine and one session factory. The app factory
creates it from settings and stores it on ``app.state`` so tests can point the
whole application at a temporary database without touching globals.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def _configure_sqlite_connection(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    # SQLite ignores foreign keys (and therefore ON DELETE CASCADE) unless
    # enabled per connection.
    cursor.execute("PRAGMA foreign_keys=ON")
    # WAL lets background ingestion write while API requests keep reading.
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


class Database:
    def __init__(self, url: str) -> None:
        is_sqlite = url.startswith("sqlite")
        connect_args = {"check_same_thread": False} if is_sqlite else {}
        self.engine: Engine = create_engine(url, connect_args=connect_args)
        if is_sqlite:
            event.listen(self.engine, "connect", _configure_sqlite_connection)
        self.session_factory = sessionmaker(
            bind=self.engine, autoflush=False, expire_on_commit=False
        )

    def create_tables(self) -> None:
        # Importing the package registers every model on Base.metadata.
        import app.models  # noqa: F401

        Base.metadata.create_all(self.engine)
        self.add_missing_columns()

    def add_missing_columns(self) -> list[str]:
        """Add nullable columns that newer versions of the models introduced.

        ``create_all`` creates missing tables but never changes existing ones,
        so a database made by an earlier version would lack new columns and
        every query on that table would fail. Nullable columns can be added
        safely; anything else needs a real migration (Alembic, planned with
        PostgreSQL support), so it is refused with a clear error instead.
        Returns the columns added, as "table.column".
        """
        inspector = inspect(self.engine)
        quote = self.engine.dialect.identifier_preparer.quote
        added: list[str] = []
        with self.engine.begin() as connection:
            for table in Base.metadata.sorted_tables:
                if not inspector.has_table(table.name):
                    continue
                existing = {column["name"] for column in inspector.get_columns(table.name)}
                for column in table.columns:
                    if column.name in existing:
                        continue
                    if not column.nullable:
                        raise RuntimeError(
                            f"The database is missing the required column {table.name}."
                            f"{column.name}. Back up and delete the data directory, or "
                            "migrate the database by hand."
                        )
                    column_type = column.type.compile(dialect=self.engine.dialect)
                    connection.execute(
                        text(
                            f"ALTER TABLE {quote(table.name)} "
                            f"ADD COLUMN {quote(column.name)} {column_type}"
                        )
                    )
                    added.append(f"{table.name}.{column.name}")
        return added

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Session for code outside a request (background tasks, scripts)."""
        session = self.session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()

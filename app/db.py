from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import Settings


@asynccontextmanager
async def get_postgres_checkpointer() -> AsyncGenerator[AsyncPostgresSaver]:
    settings = Settings()  # type: ignore
    database_url = settings.CHECKPOINTER_DATABASE_URL

    async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
        await checkpointer.setup()
        yield checkpointer


@asynccontextmanager
async def get_postgres_checkpointer_pool() -> AsyncGenerator[AsyncPostgresSaver]:
    """Checkpointer com pool de conexões, para servidores web concorrentes.

    Diferente de `get_postgres_checkpointer` (1 conexão, usado pelo CLI), aqui um
    `AsyncConnectionPool` atende várias requisições HTTP em paralelo. As `kwargs`
    (`autocommit` + `prepare_threshold=0`) e o `row_factory=dict_row` são
    requisitos do `AsyncPostgresSaver`.
    """
    settings = Settings()  # type: ignore
    database_url = settings.CHECKPOINTER_DATABASE_URL

    async with AsyncConnectionPool(
        conninfo=database_url,
        kwargs={
            'autocommit': True,
            'prepare_threshold': 0,
            'row_factory': dict_row,
        },
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        await checkpointer.setup()
        yield checkpointer

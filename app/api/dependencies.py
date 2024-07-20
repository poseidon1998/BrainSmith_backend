from distributed.client import Client
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import asyncio
from sqlalchemy.orm import Session, sessionmaker
from contextlib import asynccontextmanager

def async_session_generator(engine):
    return sessionmaker(engine, class_=AsyncSession)

def register_async_engine():
    dialect = "postgresql+asyncpg"
    user_name = "postgres"
    host = "qd3.humanbrain.in"
    port = "15432"
    db_name = "test_metadata_patch"
    password = "password"
    engine = create_async_engine(
        f"{dialect}://{user_name}:{password}@{host}:{port}/{db_name}"
    )
    return engine

@asynccontextmanager
async def get_session():
    try:
        engine = register_async_engine()
        async_session = async_session_generator(engine)
        async with async_session() as session:  # type: ignore
            yield session
    except:
        await session.rollback()
        raise
    finally:
        await session.close()

def get_dask_client() -> Client:
    from dask.distributed import Client
    client = Client(address="tcp://127.0.0.1:8786")
    return client



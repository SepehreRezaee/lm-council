import os
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection


class MongoConfig:
    def __init__(
        self,
        uri: Optional[str] = None,
        database: Optional[str] = None,
        collection: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.database = database or os.getenv("MONGO_DB", "council")
        self.collection = collection or os.getenv("MONGO_COLLECTION", "council-chats")


_client: Optional[AsyncIOMotorClient] = None


async def get_client(config: Optional[MongoConfig] = None) -> AsyncIOMotorClient:
    """Return a singleton motor client. Keeps FastAPI handlers lean (Single Responsibility)."""
    global _client
    cfg = config or MongoConfig()
    if _client is None:
        _client = AsyncIOMotorClient(cfg.uri)
    return _client


async def get_collection(
    config: Optional[MongoConfig] = None,
) -> AsyncIOMotorCollection:
    cfg = config or MongoConfig()
    client = await get_client(cfg)
    return client[cfg.database][cfg.collection]


async def ensure_indexes(config: Optional[MongoConfig] = None) -> None:
    """Ensure basic indexes exist; callable at startup."""
    coll = await get_collection(config)
    await coll.create_index("thread_id", unique=True)

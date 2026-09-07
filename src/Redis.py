import os

from redis import Redis as RedisClient


class Redis:
    def __init__(self, db=0):
        self.redis_client = RedisClient(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=False,
            db=db,
        )

    def __getattr__(self, name):
        return getattr(self.redis_client, name)
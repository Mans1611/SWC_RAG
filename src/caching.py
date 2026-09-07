import json

from redis import Redis 
import numpy as np 
from uuid import uuid4
from .Embedding import Embedding


from redis.commands.search.field import (
    VectorField,
    TextField
)

from redis.commands.search.index_definition import (
    IndexDefinition,
    IndexType
)
from redis.commands.search.query import Query
schema = (

    TextField("question"),

    VectorField(
        "embedding",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": 1024,
            "DISTANCE_METRIC": "COSINE"
        }
    )
)

from .Redis import Redis

class Cache:
    def __init__(self,embedding=None):
        self.redis_client = Redis()
        self.embedding_model = Embedding() if embedding is None else embedding 
        
  
    def create_index(self):
        self.redis_client.ft(
            "semantic_cache_idx"
        ).create_index(

            schema,

            definition=IndexDefinition(
                prefix=["cache:"],
                index_type=IndexType.HASH
            )
        )
    def add_to_cache(self,question,embedding,answer):
        cache_id = str(uuid4())
        print('************************************adding to cache************************************')
        cached_data = {
            "question" : question,
            "embedding" : np.array(
                embedding,dtype=np.float32
            ).tobytes(),
            "response" : json.dumps(answer)
        }
        self.redis_client.hset(
            f"cache:{cache_id}",
            mapping=cached_data
            )
    def search_cache(self,query,threshold=0.40):
        query_embedding = self.embedding_model.encode(query)
        print('****************************searching to cache**************************')
        
        vector = np.array(
            query_embedding,
            dtype=np.float32
        ).tobytes()
        q = Query(
            """
            *=>[KNN 1 @embedding $vector AS score]
            """
        ).sort_by("score").return_fields(
            "question",
            "response",
            "score"
        ).dialect(2)
        
        results = self.redis_client.ft(
            "semantic_cache_idx"
        ).search(
            q,
            {
                "vector": vector
            }
        )
        print(results)
        if len(results.docs) == 0: # no docs has been selected 
            return None

        result = results.docs[0]

        score = float(result.score)

        # Lower score = better
        if score > threshold:

            return None

        return json.loads(
            result.response
        )

        

if __name__ == "__main__":
    cache = Cache()
    cache.create_index()
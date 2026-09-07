from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import hashlib
from tqdm import tqdm
from .DataExtraction import DataExtraction
load_dotenv()
import os 
import time 
from qdrant_client import models
from langsmith import traceable
class VectorDatabase:
    def __init__(self,embedding=None):        
        self.qdrant_client = QdrantClient(
            url=os.getenv("QUDRANT_ENDPOINT"), 
            api_key=os.getenv("QUDRANT_API_KEY"),
            timeout=60
        )
        
        if embedding is None:
            from .Embedding import Embedding
            self.embedding = Embedding()
        else:
            self.embedding = embedding
        self.extractor = DataExtraction()
        self.batch_size = 100
    def _string_to_id(self, s: str) -> int:
        """Convert string to consistent integer ID for Qdrant"""
        return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**63)
    
    def init_collection(self, collection_name='swc_transcripts', vector_size=1024):
        """Initialize Qdrant collection if it doesn't exist"""
        try:
            self.qdrant_client.get_collection(collection_name)
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Created collection: {collection_name}")
    
    def delete_collection(self, collection_name='swc_transcripts'):
        """Delete a collection from Qdrant"""
        try:
            self.qdrant_client.delete_collection(collection_name=collection_name)
            print(f"Collection {collection_name} deleted.")
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def upsert_points(self, collection_name, points, batch_size=None):
        """Upsert points to Qdrant with batching and retry logic"""
        if batch_size is None:
            batch_size = self.batch_size
        
        total_points = len(points)
        batches = [points[i:i + batch_size] for i in range(0, total_points, batch_size)]
        
        for idx, batch in enumerate(batches):
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    self.qdrant_client.upsert(
                        collection_name=collection_name,
                        points=batch
                    )
                    print(f"Upserted batch {idx + 1}/{len(batches)} ({len(batch)} points)")
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8 seconds
                        print(f"Error upserting batch {idx + 1}: {e}")
                        print(f"Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to upsert batch {idx + 1} after {max_retries} retries: {e}")
                        return False
        
        print(f"Successfully upserted all {total_points} points to {collection_name}")
        return True
    
    def upsert_chunk(self, point_id ,  vector,payload,collection_name='swc_transcripts'):
        """Upsert a single chunk with metadata"""
        point = PointStruct(id=point_id, vector=vector, payload=payload)
        return self.upsert_points(collection_name, [point])
    
    def upsert_document(self,  chunks_data ,collection_name='swc_transcripts'):
        """Upsert entire document with multiple chunks"""
        points = []
        video_id = chunks_data[0].get('video_id')
        for idx,chunk in tqdm(enumerate(chunks_data), desc=f"Preparing chunks for {video_id}"):
            text = chunk.get("text", "")
            embedding = self.embedding.encode(text)
            point_id = self._string_to_id(f"{video_id}_{chunk.get('chunk_id', 0)}")
            
            payload = {
                "video_id": video_id,
                "chunk_id": chunk.get("chunk_id", 0),
                "start": int(chunk.get("start", 0)),
                "end": int(chunk.get("end", 0)),
                "video_title": chunk.get("video_title", ""),
                "tags": chunk.get("tags") or [],
                "text": text,
                "playlist" : chunk.get('playlist') or []
            }
            
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
            if idx+1 % 5 == 0 :
                self.upsert_points(collection_name, points)
                points=[]
        return self.upsert_points(collection_name, points)
    def upsert_documents(self,merged_dir='merged_subs'):
        self.init_collection()
        
        for file in tqdm(os.listdir(merged_dir),desc="Uploading files "):
            full_file_path = os.path.join(os.getcwd(),merged_dir,file)
            data = self.extractor.read_json(full_file_path)
            result = self.upsert_document(chunks_data=data)
            print(result)
    def upsert_video_title(self, video_id, video_title, embedding):
        """Upsert video title to video_titles collection"""
        point_id = self._string_to_id(video_id)
        payload = {
            "video_id": video_id,
            "video_title": video_title
        }
        point = PointStruct(id=point_id, vector=embedding, payload=payload)
        return self.upsert_points("video_titles", [point])
    @traceable(name = 'the search in the vectordatabase')
    def search(self,question, collection_name='swc_transcripts', limit=5,playlist_name=None, filters=None):
        """Search in a collection with optional filters"""
        query_vector = self.embedding.encode(question)
        try:
            results = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="playlist",
                            match=models.MatchValue(value=playlist_name)
                        )
                    ]
                )
            )
            return results
        except Exception as e:
            print(f"Error searching collection: {e}")
            return []
    
    def search_with_video_filter(self, query_vector, video_id,collection_name='swc_transcripts', limit=5):
        """Search within a specific video"""
        filters = Filter(
            must=[
                FieldCondition(
                    key="video_id",
                    match=MatchValue(value=video_id)
                )
            ]
        )
        return self.search(collection_name, query_vector, limit=limit, filters=filters)
    
    def search_with_playlist_filter(self, collection_name, query_vector, playlist_names, limit=5):
        """Search filtered by playlist names (requires playlist metadata in payload)"""
        if not playlist_names:
            return self.search(collection_name, query_vector, limit=limit)
        
        filters = Filter(
            must=[
                FieldCondition(
                    key="playlist",
                    match=MatchValue(value=playlist_names) if len(playlist_names) == 1 else None
                )
            ]
        ) if len(playlist_names) == 1 else None
        
        return self.search(collection_name, query_vector, limit=limit, filters=filters)
    def add_index(self,collection_name="swc_transcripts",key='playlist'):
        try:
            
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name=key,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print(f"done add the index to {collection_name} -> with field_name {key}")
            
        except Exception as e : 
            print(e)
    def format_search_results(self, results, include_rerank_score=False):
        """Format Qdrant search results into readable format"""
        formatted = []
        for result in results:
            item = {
                "text": result.payload.get("text", ""),
                "score": 1 - result.score,  # Convert Qdrant similarity to distance
                "video_title": result.payload.get("video_title", ""),
                "video_id": result.payload.get("video_id", ""),
                "start": result.payload.get("start", 0),
                "end": result.payload.get("end", 0),
                "chunk_id": result.payload.get("chunk_id", 0)
            }
            if include_rerank_score:
                item["rerank_score"] = result.payload.get("rerank_score", result.score)
            formatted.append(item)
        return formatted
    
    def get_collections(self):
        """List all collections in Qdrant"""
        try:
            collections = self.qdrant_client.get_collections()
            return collections
        except Exception as e:
            print(f"Error getting collections: {e}")
            return None
        
    def get_chunks_by_video_id(
    self,
    video_id,
    collection_name="swc_transcripts",
    limit=3
):
        """Retrieve all chunks belonging to a specific video_id"""

        filters = Filter(
            must=[
                FieldCondition(
                    key="video_id",
                    match=MatchValue(value=video_id)
                )
            ]
        )

        points, _ = self.qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=filters,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        return points
    
    
if __name__ =='__main__':
    qdrant = VectorDatabase()
    qdrant.add_index()
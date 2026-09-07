from .DataExtraction import DataExtraction
from .Chunking import Chunking
from sentence_transformers import CrossEncoder, SentenceTransformer
import os 
from tqdm import tqdm
from .schemas.SubtitleDict import SubtitleDict
from src.Qudrant import VectorDatabase
class Embedding: 
    def __init__(self, embedding_model=None):
        if embedding_model is None:
            self.embedding_model = SentenceTransformer("BAAI/bge-m3")
            self.reranker = CrossEncoder("BAAI/bge-reranker-base")  
        
        self.vector_db = VectorDatabase(embedding=self.embedding_model)
        self.extractor = DataExtraction()
        self.chunking = Chunking()
        
    def encode(self, txt: str): 
        """Encode text to embedding vector"""
        return self.embedding_model.encode(txt, normalize_embeddings=True).tolist()
    
    def delete_collection(self, collection_name='swc_transcripts'):
        """Delete collection via VectorDatabase"""
        return self.vector_db.delete_collection(collection_name)
    
    def store_document(self, transcript_data: SubtitleDict, video_id: str) -> None:
        """Store document with automatic chunking and embedding"""
        documents = self.chunking.create_chunks(transcript_data)
        points = []

        for doc in tqdm(documents, desc=f"Preparing chunks for {video_id}"):
            text = doc.page_content
            metadata = doc.metadata
            
            # Create embedding
            embedding = self.embedding_model.encode(text, normalize_embeddings=True).tolist()
            point_id = self.vector_db._string_to_id(f"{video_id}_{metadata['chunk_id']}")

            payload = {
                "video_id": video_id,
                "chunk_id": metadata["chunk_id"],
                "start": int(metadata["start"]),
                "end": int(metadata["end"]),
                "video_title": metadata['video_title'],
                "text": text
            }
            
            points.append(type('PointStruct', (), {'id': point_id, 'vector': embedding, 'payload': payload})())

        # Upsert to Qdrant via VectorDatabase
        self.vector_db.upsert_points("swc_transcripts", points)
    
    def store_video_title(self, video_id: str, video_title: str):
        """Store video title embedding"""
        embedding = self.embedding_model.encode(video_title, normalize_embeddings=True).tolist()
        self.vector_db.upsert_video_title(video_id, video_title, embedding)
    
    def store_videos_title(self, meta_data='swc_videos.json'):
        """Store all video titles from metadata file"""
        videos = self.extractor.read_json(meta_data)
        for video in tqdm(videos, desc="Uploading video titles"):
            self.store_video_title(video_id=video["video_id"], video_title=video['video_title'])
    
    def store_documents(self, merged_subs_path='merged_subs'):
        """Store all documents from directory"""
        for file in tqdm(os.listdir(merged_subs_path), desc=f"Uploading files"):
            full_path = os.path.join(merged_subs_path, file)
            self.store_one_video_in_database(full_path)
            print(f"File {file} completed.")
    
    def store_one_video_in_database(self, sub_path):
        """Store single video chunks"""
        from qdrant_client.models import PointStruct
        points = []
        merged_chunks = self.extractor.read_json(sub_path)
        video_id = sub_path.replace("transcribe_", "").replace(".json", "")
        
        for chunk in tqdm(merged_chunks, desc=f"Uploading chunks for {video_id}"):
            text = chunk["text"]
            embedding = self.embedding_model.encode(text, normalize_embeddings=True).tolist()
            point_id = self.vector_db._string_to_id(f"{video_id}_{chunk['chunk_id']}")

            payload = {
                "video_id": video_id,
                "chunk_id": chunk["chunk_id"],
                "start": int(chunk["start"]),
                "end": int(chunk["end"]),
                "video_title": chunk['video_title'],
                "tags": [] if chunk["tags"] is None else chunk["tags"],
                "text": text
            }
            
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
        
        self.vector_db.upsert_points("swc_transcripts", points)
    
    def store_vectors(self, input_dir='merged_subs'):
        """Store all vectors from directory"""
        jsons_files = os.listdir(input_dir)
        for file_path in tqdm(jsons_files, desc="Uploading files"):
            full_path = os.path.join(input_dir, file_path)
            self.store_one_video_in_database(full_path)
    
    def retrieve_video_title(self, question, top_k=3):
        embed_query = self.embedding_model.encode(question).tolist()
        
        results = self.qdrant_client.search(
            collection_name="video_titles",
            query_vector=embed_query,
            limit=top_k
        )
        
        retrieved_chunks = []
        for result in results:
            retrieved_chunks.append({
                "video_title": result.payload["video_title"],
                "score": 1 - result.score,  # Qdrant similarity to distance
                "video_id": result.payload["video_id"]
            })
        
        retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x['score'])
        return retrieved_chunks
    def retrieve_chunks_inside_video(self, query, video_id, top_k=5):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        results = self.qdrant_client.search(
            collection_name="swc_transcripts",
            query_vector=query_embedding,
            limit=top_k,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchValue(value=video_id)
                    )
                ]
            )
        )
        
        retrieved_chunks = []
        for result in results:
            retrieved_chunks.append({
                "text": result.payload["text"],
                "score": 1 - result.score,
                "video_title": result.payload["video_title"],
                "video_id": result.payload["video_id"],
                "start": result.payload["start"],
                "end": result.payload["end"]
            })
        
        return retrieved_chunks     
    def retrieve(self, question, top_k=3, video_id=None):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        embed_query = self.embedding_model.encode(question).tolist()
        
        # Build filter if video_id is specified
        query_filter = None
        if video_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchValue(value=video_id)
                    )
                ]
            )
        
        results = self.qdrant_client.search(
            collection_name="swc_transcripts",
            query_vector=embed_query,
            limit=top_k,
            query_filter=query_filter
        )

        retrieved_chunks = []
        for result in results:
            retrieved_chunks.append({
                "text": result.payload["text"],
                "score": 1 - result.score,
                "video_title": result.payload["video_title"],
                "video_id": result.payload["video_id"],
                "start": result.payload["start"],
                "end": result.payload["end"]
            })
        
        # Rerank with CrossEncoder
        pairs = [[question, f"فيديو: {res['video_title']} | المحتوى: {res['text']}"] for res in retrieved_chunks]
        scores = self.reranker.predict(pairs)
        for i, res in enumerate(retrieved_chunks):
            res["rerank_score"] = scores[i]
        
        final_sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        return final_sorted_chunks, embed_query
    
if __name__ == '__main__' : 
    embedding = Embedding()
    print(embedding.encode("عرفني على المعادلات التفاضلية "))
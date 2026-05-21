import chromadb
from .DataExtraction import DataExtraction
from sentence_transformers import SentenceTransformer
from .Chunking import Chunking
import os 
from tqdm import tqdm
from .schemas.SubtitleDict import SubtitleDict
from sentence_transformers import CrossEncoder

class Embedding: 
    def __init__(self,embedding_model = "BAAI/bge-m3"):
        self.embedding_model = SentenceTransformer(embedding_model)
        
        self.client = chromadb.PersistentClient(path="./vectordb")
        
        self.collection = self.client.get_or_create_collection(
            name = 'swc_transcripts'
        )
        # this is for the the video_title embedding 
        self.video_collection = self.client.get_or_create_collection(
            name="vide_title"
        )
    
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")
        self.extractor = DataExtraction()
        self.chunking = Chunking()
        
        
        
    def encode(self,txt:str): 
        return len(self.embedding_model.encode(txt).tolist())
    def delete_collection(self,collection_name='swc_transcripts'):
        try : 
            self.client.delete_collection(name=collection_name)
            print("collection is being removed.")
            return True
        except Exception as e : 
            print("Collection is not reamoved : ",e)
            return False 
        
    def store_document(self,transcript_data:SubtitleDict,video_id:str)-> None:
        documents = self.chunking.create_chunks(transcript_data)
        ids = []
        embeddings = []
        metadatas = []
        texts = []

        for doc in tqdm(documents,desc=f"Uploading chunk."):

            text = doc.page_content

            metadata = doc.metadata
            # Create embedding
            embedding = self.embedding_model.encode(
                text,
                normalize_embeddings=True
            ).tolist()

            ids.append(
                f"{video_id}_{metadata['chunk_id']}"
            )

            embeddings.append(embedding)

            texts.append(text)

            metadatas.append({
                "video_id": video_id,
                "chunk_id": metadata["chunk_id"],
                "start": int(metadata["start"]),
                "end": int(metadata["end"]),
                "video_title" : metadata['video_title']
            })

        # Upload to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"Stored {len(texts)} chunks")
    
    def store_video_title(self,video_id:str,video_title:str):

        embedding = self.embedding_model.encode(
            video_title,
            normalize_embeddings=True
        ).tolist()

        self.video_collection.add(
            ids=[video_id],
            embeddings=[embedding],
            documents=[video_title],
            metadatas=[{
                "video_id": video_id,
                "video_title": video_title
            }]
        )     
    def store_videos_title(self,meta_data='swc_videos.json'):
        videos = self.extractor.read_json(meta_data)
        for video in tqdm(videos,desc="Uploading video"):
            self.store_video_title(video_id=video["video_id"],video_title=video['video_title'])
            
            
             
              
    def store_documents(self,merged_subs_path='merged_subs'):
        for file in tqdm(os.listdir(merged_subs_path),desc=f"Uploading file "):
            full_path  = os.path.join(merged_subs_path,file)
            data = self.extractor.read_json(full_path)
            video_id = file.replace("transcribe_", "").replace(".json", "")
            self.store_document(data,video_id)
            print(f" this is the file {file} is done .")
            
    def store_one_video_in_database(self,sub_path):
        documents = []
        embeddings = []
        ids = []
        metadatas = []
        merged_chunks = self.extractor.read_json(sub_path)
        video_id = sub_path.replace("transcribe_", "").replace(".json", "")
                
        for chunk in tqdm(merged_chunks,desc=f"uploading cunks for-- {video_id}-- "):

            text = chunk["text"]

            embedding = self.embedding_model.encode(text).tolist()

            documents.append(text)

            embeddings.append(embedding)

            ids.append(f"{video_id}_{chunk['chunk_id']}")

            metadatas.append({
                "video_id": video_id,
                "chunk_id": chunk["chunk_id"],
                "start": int(chunk["start"]),
                "end": int(chunk["end"]),
                "video_title" : chunk['video_title']
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    def store_vectors(self,input_dir):
        ''' 
            storing each file in 

            input_dir : the directory of the merged transcripts 
        '''
        jsons_files = os.listdir(input_dir)
        for file_path in tqdm(jsons_files,desc="uploading files ") : 
            full_path = os.path.join(input_dir,file_path)
            self.store_one_video_in_database(full_path)
    
    def retrieve_video_title(self,question,top_k=3):
        
        embed_query = self.embedding_model.encode(question).tolist()
        
        results = self.video_collection.query(
            query_embeddings=[embed_query],
            n_results=top_k
        )
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        retrieved_chunks = []
        print(results)
        for doc, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            retrieved_chunks.append({
                "video_title": doc,
                "score": distance,
                "video_id": metadata["video_id"]
            })
        retrieved_chunks = sorted(retrieved_chunks,key = lambda x : x['score'])   
        return retrieved_chunks
    def retrieve_chunks_inside_video(self,query,video_id,top_k=5):

        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        results = self.chunk_collection.query(
            query_embeddings=[query_embedding],

            n_results=top_k,

            where={
                "video_id": video_id
            }
        )

        return results     
    def retrieve(self,question,top_k=3,video_id=None):
        
        embed_query = self.embedding_model.encode(question).tolist()
        results = None
        if video_id :
            results = self.collection.query(
                query_embeddings=[embed_query],
                n_results=top_k,
                where={
                "video_id": video_id
            }
            )
        else : 
            results = self.collection.query(
                query_embeddings=[embed_query],
                n_results=top_k
            )
        print(f'------------- results ({__name__})-----------------------')
        print(results)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        retrieved_chunks = []
        for doc, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):
            retrieved_chunks.append({
                "text": doc,
                "score": distance,
                "video_title" : metadata["video_title"],
                "video_id": metadata["video_id"],
                "start": metadata["start"],
                "end": metadata["end"]
            })
        
        pairs = [[question, f"فيديو: {res['video_title']} | المحتوى: {res['text']}"] for res in retrieved_chunks]
        scores = self.reranker.predict(pairs)
        for i, res in enumerate(retrieved_chunks):
            res["rerank_score"] = scores[i]
        final_sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        return final_sorted_chunks
    
if __name__ == '__main__' : 
    embedding = Embedding()
    print(embedding.retrieve_video_title("عرفني على المعادلات التفاضلية "))
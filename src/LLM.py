from .Embedding import Embedding
from google import genai
from dotenv import load_dotenv
import os 
from .schemas.RAGOutput import RAGOutput
from google.genai.types import GenerateContentConfig, GenerateContentConfigOrDict
import json 
load_dotenv()

import ollama

class LLM:
    def __init__(self,model_name='gemini-2.5-flash'):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.embedding = Embedding()
    def build_prompt(self,user_question:str,video_id=None,retrieved_chunks=None):
        if retrieved_chunks : 
            retrieved_chunks = self.embedding.retrieve(question=user_question,video_id=video_id)

        context = ""

        for idx, chunk in enumerate(retrieved_chunks):

            context += f"""
                Chunk {idx + 1}:
                Video ID: {chunk['video_id']}
                Start: {chunk['start']}
                End: {chunk['end']}
                Content:
                {chunk['text']}
                """

        prompt = f"""
            You are an Arabic educational Math assistant, related To Math Channel called SWC,
            which is a speacilized channel for explaning Math Concepts.

            Your task is:
            1. Answer the user's question briefly and clearly.
            2. Use ONLY the retrieved context below.
            3. Mention that the concept is explained in the retrieved video.
            4. Return ONLY valid JSON.
            5. Do NOT return markdown.
            6. Do NOT return explanations outside JSON.
            7. Choose the BEST and MOST relevant chunk.

            User Question:
            {user_question}

            Retrieved Context:
            {context}

            Return JSON in this EXACT schema:

            {{
                "video_id": "...",
                "generated_text": "...",
                "start": 0.0,
                "end": 0.0
            }}

            Rules:
            - generated_text must be in Arabic.
            - generated_text should briefly explain the concept.
            - generated_text should mention that the explanation exists in the video.
            - video_id/start/end must come from the BEST retrieved chunk only.
            """

        return prompt
    
    def build_router_prompt(self,question):

        prompt = f"""
            You are a retrieval routing agent.

            Your task:
            Decide which retrieval strategy is best.

            Available strategies:

            1. video_title
            - Use when the question is asking about
            a broad educational topic that may match
            a video title.

            2. chunk_search
            - Use when the question is highly specific
            and may require detailed transcript retrieval.

            Return ONLY valid JSON.

            Schema:
            {{
                "strategy": "video_title"
            }}

            OR

            {{
                "strategy": "chunk_search"
            }}

            Question:
            {question}
            """
        return prompt
    def select_strategy(self,question:str):
        response = self.client.models.generate_content(
            model = self.model_name,
            contents={"text":self.build_router_prompt(question=question)},
            config={
                "temperature": 0,
                "response_mime_type": "application/json"
            }
        )

        return json.loads(
            response.text
        )["strategy"]
        
    def retrive_by_video_title(self,query,threshold = 0.7):
        query_embedding = self.embedding.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        results = self.embedding.video_collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )

        distance = results["distances"][0][0]

        if distance < threshold:

            return {
                "found": True,
                "video_id":
                    results["metadatas"][0][0]["video_id"],
                "score": distance
            }

        return {
            "found": False
        }
    
    def generate(self,user_question:str):
        
        
        retrieved_chunks = None
            
        title_chunks = (
            self.retrive_by_video_title(user_question)
        )
        
        print('---------------------title_chunks------------------------')
        print(title_chunks)  
        print('---------------------title_chunks------------------------')
        if title_chunks['found']: 
            retrieved_chunks = self.embedding.retrieve(
                question=user_question,
                video_id=title_chunks['video_id']
            )
        else : 
            retrieved_chunks = self.embedding.retrieve(
                question=user_question
            )
        
        
        prompt = self.build_prompt(
            user_question=user_question,
            retrieved_chunks = retrieved_chunks)
        try:
            result = self.client.models.generate_content(
                model = self.model_name,
                contents={"text":prompt},
                config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                    "response_schema": RAGOutput
                }
            )
            return result.text
        except Exception as e : 
            print('****'*20)
            print(e)
            return {
                "llm_response":e,
                "start" : None,
                'end' : None,
                "video_id" : None,
                
            }
        
if __name__ == "__main__" : 
    llm = LLM(model_name="gemini-2.5-flash")
    # print('----'*30)
    # print(llm.build_prompt("هو ايه عامل التكامل "))
    # print('----'*30)

    print(llm.generate("ممكن شرح لاختبار التكامل في المتسلسلات"))
    
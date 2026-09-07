from src.Qudrant import VectorDatabase
from .ChatMemory import ChatMemory
from .Embedding import Embedding
from google import genai
from dotenv import load_dotenv
import os 
from .schemas.RAGOutput import RAGOutput
from google.genai.types import GenerateContentConfig, GenerateContentConfigOrDict
import json 
load_dotenv()
from .LightLLM import LightLLM
from langsmith import traceable
from .schemas.GenereteRequest import GenereteRequest
class LLM:
    def __init__(self,model_name='gemini-2.5-flash',embedding=None,session_id=None):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.embedding = Embedding() if embedding is None else embedding
        self.light_llm = LightLLM()
        self.qdrant = VectorDatabase(embedding=self.embedding)
        self.session_id = session_id
        self.chat_history = ChatMemory()
    # def set_embedding_model(self,request):
    #     self.embedding =  Embedding(request.app.state.embedding_model,request.app.state.reranker)

    def build_prompt(self, user_question: str, video_id=None, retrieved_chunks=None, messages=None):


        context = ""
        history = "\n".join(
            f"{message.get('role', 'unknown')}: {message.get('content', '')}"
            for message in (messages or [])
        )

        for idx, chunk in enumerate(retrieved_chunks):

            context += f"""
                Chunk {idx + 1}:
                Start: {chunk['start']}
                End: {chunk['end']}
                Video_id : {chunk['video_id']}
                Content:
                {chunk['text']}
                """

        prompt = f"""
            You are an Arabic educational Math assistant, related To Math Channel called SWC,
            which is a speacilized channel for explaning Math Concepts.

            Your task is:
            1. Answer the user's question clearly.
            2. Use ONLY the retrieved context below.
            3. Mention that the concept is explained in the retrieved video.
            4. Return ONLY valid JSON.
            5. Do NOT return markdown.
            6. Do NOT return explanations outside JSON.
            7. Choose the BEST and MOST relevant chunk that is related to the user question, based on Two things : 
                a - video_title, and most related to the user question 
                b - the text in each chunk and whether it is related to the user question.

            User Question:
            {user_question}

            Previous Conversation:
            {history}

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
        
    # def retrive_by_video_title(self,query,threshold = 0.7):
    #     query_embedding = self.embedding.embedding_model.encode(
    #         query,
    #         normalize_embeddings=True
    #     ).tolist()

    #     results = self.embedding.video_collection.query(
    #         query_embeddings=[query_embedding],
    #         n_results=1
    #     )

    #     distance = results["distances"][0][0]

    #     if distance < threshold:

    #         return {
    #             "found": True,
    #             "video_id":
    #                 results["metadatas"][0][0]["video_id"],
    #             "score": distance
    #         }

    #     return {
    #         "found": False
    #     }
    def pretify_chunks(self,chunks=[]):
        new_chunks = []
        print(chunks)
        for res in chunks : 
            new_chunks.append(res.payload)
        return new_chunks
    @traceable(name='Traking the LLM generete')
    def generate(self,request:GenereteRequest):
        retrieved_chunks = None
        try:
            messages = self.chat_history.get_messages(session_id=request.session_id)
            intent_json = json.loads(self.light_llm.generate(user_question=request.user_question,messages = messages))
        
            
            print('--------'*10)
            print(intent_json)
            print('--------'*10)
            print(intent_json.get('video_playlist'))
            
            retrieved_chunks = self.qdrant.search(
                question=request.user_question,
                playlist_name=intent_json.get('video_playlist',[])[0],
                limit=10
            )
            retrieved_chunks = self.pretify_chunks(chunks=retrieved_chunks.points)
            print('----------------- retrieved Chunks ---------------------------')
            print(retrieved_chunks)
            print('----------------- retrieved Chunks ---------------------------')
            print('----------------- previous Messagees ---------------------------')
            print(messages)
            print('----------------- previous Messagees ---------------------------')
            prompt = self.build_prompt(
                user_question=request.user_question,
                retrieved_chunks = retrieved_chunks,
                messages = messages
                )
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
                return result.text,[]
            except Exception as e : 
                print('****'*20)
                print(e)
                return {
                    "llm_response":e,
                    "start" : None,
                    'end' : None,
                    "video_id" : None,
                    
                },None
        except Exception as e : 
            print("the error is ya mans : ")
            print(e)
            return {
                    "llm_response":e,
                    "start" : None,
                    'end' : None,
                    "video_id" : None,
                    
                },None
            
if __name__ == "__main__" : 
    llm = LLM(model_name="gemini-2.5-flash")
    # print('----'*30)
    # print(llm.build_prompt("هو ايه عامل التكامل "))
    # print('----'*30)

    print(llm.generate("ممكن شرح لاختبار التكامل في المتسلسلات"))
    
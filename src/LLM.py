from .Embedding import Embedding
from google import genai
from dotenv import load_dotenv
import os 
from .schemas.RAGOutput import RAGOutput
from google.genai.types import GenerateContentConfig, GenerateContentConfigOrDict 
load_dotenv()
class LLM:
    def __init__(self,model_name='gemini-2.5-flash'):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.embedding = Embedding()
    def build_prompt(self,user_question:str):
        retrieved_chunks = self.embedding.retrieve(question=user_question)
        print('-------------------------- retrived ------------------------------------')
        print(retrieved_chunks)
        print('-------------------------- retrived ------------------------------------')

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
    def generate(self,user_question:str):
        
        prompt = self.build_prompt(user_question=user_question)
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
    
if __name__ == "__main__" : 
    llm = LLM(model_name="gemini-2.5-flash")
    # print('----'*30)
    # print(llm.build_prompt("هو ايه عامل التكامل "))
    # print('----'*30)

    print(llm.generate("Explain to me Integrating factor?"))
    
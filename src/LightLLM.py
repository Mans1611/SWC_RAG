from src.Qudrant import VectorDatabase

from .Embedding import Embedding
from google import genai
from dotenv import load_dotenv
import os 
from .schemas.Classifier_Output import ClassifierOutput 
from google.genai.types import GenerateContentConfig, GenerateContentConfigOrDict
import json 
from .DataExtraction import DataExtraction
load_dotenv()
from langsmith import traceable
class LightLLM:
    def __init__(self,model_name='gemini-2.5-flash'):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.extractor = DataExtraction()

        
        
    def build_prompt(self,user_question:str,messages):
        playlist_names = self.extractor.get_all_playlist_names()

        prompt = f"""
            You are an intent classifier for the SWC YouTube channel. You will receive:
            1) the user question,
            2) the exact playlist names for the channel,
            3) The Chat History, so you can depend on it if the semantic of the message was unclear
            4) the requirement to return only valid JSON with no additional commentary.

            Your task:
            - Determine which playlist are relevant to the user request and return them in `"video_playlist"`.
            - Determine which playlists should be excluded from retrieval and return them in `"exclude_playlist"`.
            - Determine the overall query type and return it as `"query_type"`, where the value must be exactly either:
            - `"question"` when the user is asking for examples, problems, exercises, solutions, or specific task-based help.
            - `"explanation"` when the user is asking for concept explanation, definitions, theory, intuition, or understanding.
            - Determine whether the user wants just a video or a full playlist or a course, the key is the media_type, and it whether a video, or course.
            Important rules:
            - Use only the exact playlist names provided below.
            - Do not modify the playlist names: they must match exactly.
            - Do not invent new playlist names.
            - Do not output anything but valid JSON.
            - Do not exclude the rest of the playlist names, so if you pick the video playlist, do not exclude the rest, You cam just exclude the irrlevent playlists. 
            - `"video_playlist"` and `"exclude_playlist"` must always be arrays.
            - `"query_type"` must be exactly `"question"` or `"explanation"`.

            Playlist names:
                {playlist_names}

            JSON schema to return:
            {{
            "video_playlist": [],
            "exclude_playlist": [],
            "query_type": "",
            "media_type": ""
            
            }}

            Example logic:
            - If the user asks for "شرح" or "ما هو" or "لماذا" => `"query_type": "explanation"`.
            - If the user asks for "حل" or "مثال" or "سؤال" or "تمرين" => `"query_type": "question"`.
            - If the user mentions a playlist topic clearly, include that exact playlist in `"video_playlist"`.
            - Add other playlists to `"exclude_playlist"` when they do not match the user intent.
            Previous Chat History : 
            {messages}
            Now classify this request and return only the JSON result.
            User Question : {user_question}
            """

        return prompt
    @traceable(name='get playlist name')
    def generate(self,user_question,messages):
        prompt = self.build_prompt(user_question=user_question,messages=messages)
        try:
            result = self.client.models.generate_content(
                model = self.model_name,
                contents={"text":prompt},
                config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                    "response_schema": ClassifierOutput    
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
    llm = LightLLM()
    # print('----'*30)
    # print(llm.build_prompt("هو ايه عامل التكامل "))
    # print('----'*30)
    print(llm.generate("ايه هو عامل التكامل "))
    
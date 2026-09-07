from pydantic import BaseModel



class GenereteRequest(BaseModel) : 
    user_question : str 
    session_id : str
    
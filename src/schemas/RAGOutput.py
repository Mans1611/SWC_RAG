from pydantic import BaseModel


class RAGOutput(BaseModel):
    video_id : str
    llm_output : str 
    start : float 
    end : float
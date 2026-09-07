from pydantic import BaseModel
from typing import List

class ClassifierOutput(BaseModel) : 
    video_playlist: List[str] 
    exclude_playlist: List[str]
    query_type: str
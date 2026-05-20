from pydantic import BaseModel


class SubtitleDict(BaseModel):

    chunk_id: int
    text: str
    start: int
    end: int
    video_title : str
from fastapi import APIRouter
from ..schemas.GenereteRequest import GenereteRequest
from src.LLM import LLM 

llm_router = APIRouter() 
llm = LLM()

@llm_router.post('/generete/')
async def generete_answer(request:GenereteRequest):
    return llm.generate(request.user_question)
    
from fastapi import APIRouter
from ..schemas.GenereteRequest import GenereteRequest
from src.LLM import LLM
from ..utils.rag_evaluator import rag_evaluator
import json

llm_router = APIRouter() 
llm = LLM()

@llm_router.post('/generete/')
async def generete_answer(request:GenereteRequest):
    
    result = llm.generate(request.user_question)
    
    # Parse the result to extract information for evaluation
    try:
        print(f'-----------result ({__name__})------------------')
        print(result)
        result_data = json.loads(result)
        
        # Extract fields from the LLM response
        llm_response = result_data.get('llm_output', '')
        video_id = result_data.get('video_id', '')
        start = result_data.get('start', 0)
        end = result_data.get('end', 0)
        
        # Save to evaluation CSV
        rag_evaluator.add_record(
            user_question=request.user_question,
            llm_response=llm_response,
            video_id=video_id,
            start=start,
            end=end,
            true_answer=""  # Can be filled later for evaluation
        )
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing LLM response: {e}")
    
    return result 
    
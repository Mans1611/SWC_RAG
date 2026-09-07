from fastapi import APIRouter
from ..schemas.GenereteRequest import GenereteRequest
from src.LLM import LLM
from ..utils.rag_evaluator import rag_evaluator
import json
from src.caching import Cache
from src.Embedding import Embedding
from src.ChatMemory import ChatMemory
from langsmith import traceable
llm_router = APIRouter() 
embedding = Embedding()
llm = LLM(embedding=embedding)
cache = Cache(embedding=embedding)
chat_memory = ChatMemory()


@llm_router.post('/generete/')
@traceable(name="rag_endpoint")
async def generete_answer(request:GenereteRequest):
    cache_result = cache.search_cache(request.user_question)
    if cache_result:
        print("✅✅✅✅✅✅✅✅✅✅✅✅✅ cache Hit ✅✅✅✅✅✅✅✅✅✅✅✅")
        result_data = cache_result
    else:
        result, embed_query = llm.generate(request)
        result_data = result
    
    # Parse the result to extract information for evaluation
    try:
        print("❌❌❌❌❌❌❌❌❌ cache MiSS ❌❌❌❌❌❌❌❌❌❌")
        if isinstance(result_data, str):
            result_data = json.loads(result_data)
        print('-----------------------result ya mans------------------------------')
        print(result_data)
        #cache.add_to_cache(request.user_question,embedding=embed_query,answer=result)
        # Extract fields from the LLM response
        llm_response = result_data.get('llm_output', '')
        video_id = result_data.get('video_id', '')
        start = result_data.get('start', 0)
        end = result_data.get('end', 0)

        chat_memory.add_messages(
            request.session_id,
            [
                {"role": "human", "content": request.user_question},
                {
                    "role": "ai",
                    "content": llm_response,
                    "video_id": video_id,
                    "start": start,
                    "end": end,
                },
            ],
        )
        
        # Save to evaluation CSV
        rag_evaluator.add_record(
            user_question=request.user_question,
            llm_response=llm_response,
            video_id=video_id,
            start=start,
            end=end,
            true_answer=""  # Can be filled later for evaluation
        )
       
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return {
            "llm_output" : "",
            "start":0,
            "end":0,
            "video_id" : None
            
        }
    print(type(result_data))
    print('--------------------')
    print((type(result)))
    return result_data
    

import json
from typing import Any, Dict, List

from .Redis import Redis


class ChatMemory:
    def __init__(self):
        self.memory_client = Redis(db=1)

    def add_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        try:
            self.memory_client.rpush(
                f"chat_history:{session_id}",
                json.dumps(message, ensure_ascii=False),
            )
            return True
        except Exception as error:
            print(f"the error is {error}")
            return False 

    def add_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> bool:
        return all(self.add_message(session_id, message) for message in messages)

    def get_messages(self, session_id: str, n: int = 10) -> List[Dict[str, Any]]:
        try:
            messages = self.memory_client.lrange(
                f"chat_history:{session_id}",
                -n,
                -1,
            )
            return [json.loads(message) for message in messages]
        except Exception as error:
            print(error)
            return []


if __name__ == '__main__' : 
    chat = ChatMemory()
    session_id = "mans_1611"
    chat.add_messages(
        session_id=session_id,
        messages=[
            {
                "role": "human",
                "content": "hiow are you doing ahmed",
            },
             {
                "role": "ai",
                "content": "I am gemini, i dont have feelings, like ahmed.",
            },
        ],
    )
    
    messages = (chat.get_messages(session_id=session_id))
    
    print(messages)
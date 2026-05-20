from src.Embedding import Embedding

from src.DataExtraction import DataExtraction
from src.Embedding import Embedding
from src.LLM import LLM

llm = LLM()



question  = "عرفني كدا على معامل التكامل"
embedding = Embedding()

print(llm.generate(user_question=question))

from src.Embedding import Embedding

from src.DataExtraction import DataExtraction
from src.Embedding import Embedding
from src.LLM import LLM
from src.Qudrant import VectorDatabase

# extractor = DataExtraction()
# extractor.add_playlist_to_chunks()

# extractor.add_playlist(
#     playlist_json_file='videos_playlist_title.json',
#     videos_file='swc_videos_metadata.json'
# )
# print(extractor.get_all_playlist_names())


qdrant = VectorDatabase()
# qdrant.delete_collection()
# qdrant.init_collection()
# qdrant.upsert_documents()
question = "اشرحلي ازاي ممكن استخدم اختبار التكامل" 
playlist = "Sequences & Series | المتتاليات و المتسلسلات"
    
results = qdrant.search(question=question,playlist_name=playlist).points
print(results)
chunks = []
for res in results : 
    print(res.payload)
    chunks.append(res.payload)
    print('====='*10)
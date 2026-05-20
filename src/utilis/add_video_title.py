import os 
import json
from ..DataExtraction import DataExtraction


def add_video_title_2_chunks(all_video_metadata_path='./youtube_videos.json',subs_dir='swc_subs'):
     
    extractor = DataExtraction()
    subs_files = os.listdir(subs_dir)
    all_videos_metadata = extractor.read_json(all_video_metadata_path)
   
    for file in subs_files : 
        video_id = file.replace("transcribe_", "").replace(".json", "")
        video_title = None    
        for video in all_videos_metadata : 
            if video['video_id'] == video_id : 
                video_title = video['video_title']
        full_path = os.path.join(subs_dir,file)
        chunks = extractor.read_json(full_path)
        for chunk in chunks : 
            chunk['video_title'] = video_title
        with open(full_path,'w') as f : 
            json.dump(chunks,f,ensure_ascii=False,indent=4)
                      

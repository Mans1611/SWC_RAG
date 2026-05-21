import json
from youtube_transcript_api.proxies import WebshareProxyConfig
import yt_dlp
import os 
import logging 
from tqdm import tqdm

logger =  logging.getLogger(__name__)
class DataExtraction:
    
    def __init__(self,channel_url=None,output_file_name='youtube_videos'):
        self.channel_url = channel_url
        self.output_file_name = output_file_name
        
        
    def extract_videos_metadata(self,output_file_name='youtube_videos'):
        self.output_file_name = f"{output_file_name}.json"
            
        ydl_opts = {
            "extract_flat": True,
            "quiet": True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                self.channel_url,
                download=False
            )

            with open(self.output_file_name, "w", encoding="utf-8") as f:
                cleaned = []
                for data in info['entries']:
                    for entry in data["entries"]:
                        cleaned.append({
                            "video_title" : entry['title'],
                            "video_url" : entry['url'],
                            "video_duration" : entry["duration"] if 'shorts' not in entry['url'].split('/') else None,
                            "video_id" : entry['url'].split('=')[-1]
                        })
                json.dump(cleaned, f, ensure_ascii=False, indent=4)
    
    def already_downloaded(self,output_dir:str):
        downloaded_ids = set()
        for file_name in os.listdir(output_dir):
            if file_name.startswith("transcribe_") and file_name.endswith(".json"):
                video_id = file_name.replace("transcribe_", "").replace(".json", "")
                downloaded_ids.add(video_id)
        return downloaded_ids
    
    def extract_subtitle(self,output_dir='swc_subs'):
        from youtube_transcript_api import YouTubeTranscriptApi
        
        full_output_dir = os.path.join(os.getcwd(),output_dir)
        os.makedirs(full_output_dir,exist_ok=True)
        downloaded_ids = self.already_downloaded(full_output_dir)
        print('------------'*10)
        print((downloaded_ids))
        print('------------'*10)
        
        with open(f"{self.output_file_name}.json",'r',encoding='utf-8') as f : 
            videos_metadata = json.load(f)
            
        ytt_api = YouTubeTranscriptApi()
        
        
        for _,video in tqdm(enumerate(videos_metadata),desc = "downloadig video") : 
            if video['video_id'] in downloaded_ids : continue
            
            output = ytt_api.fetch(video['video_id'],languages=['ar'])
            video_metadata = [] 
            for idx,transcribe in enumerate(output) :
                video_metadata.append({
                    "chunk_id" : idx,
                    "text" : transcribe.text,
                    "start" : transcribe.start,
                    "end" : transcribe.start + transcribe.duration
                })
            outputfile = os.path.join(os.getcwd(),output_dir,f'transcribe_{video['video_id']}.json')
            with open(outputfile,'w',encoding = 'utf-8') as f : 
                json.dump(video_metadata,f,ensure_ascii=False, indent=4)
    
    def __merge_transcript_chunks(self, transcript_data, chunk_size=5):
        merged_data = []

        for i in range(0, len(transcript_data), chunk_size):

            # Take every 4 objects
            chunk_group = transcript_data[i:i + chunk_size]

            # Concatenate text
            merged_text = " ".join(
                chunk["text"] for chunk in chunk_group
            )

            # Create merged object
            merged_object = {
                "chunk_id": i // chunk_size,
                "text": merged_text,
                "start": chunk_group[0]["start"],   # start of first object
                "end": chunk_group[-1]["end"],       # end of last object
                "video_title" : chunk_group[-1]['video_title']
            }

            merged_data.append(merged_object)

        return merged_data
    
    def read_json(self,input_dir): 
        with open(input_dir,'r',encoding='utf-8') as f : 
            data = json.load(f)
        return data 
    
    def make_merge(self,input_dir='swc_subs',output_dir='merged_subs',chunk_size=5):
        output_dir = os.path.join(os.getcwd(),output_dir)
        os.makedirs(output_dir,exist_ok=True)
        for file in os.listdir(input_dir):
            full_file_path = os.path.join(input_dir,file)
            data = self.read_json(full_file_path)
            merged = self.__merge_transcript_chunks(data,chunk_size=chunk_size)
            new_file = os.path.join(output_dir,file)
            with open(f"{new_file}",'w',encoding='utf-8') as f : 
                json.dump(merged,f,ensure_ascii=False, indent=4)
            logger.info(f"Done with file {full_file_path}.")
            
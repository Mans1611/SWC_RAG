import json
from youtube_transcript_api.proxies import WebshareProxyConfig
import yt_dlp
import os 
import logging 
from tqdm import tqdm
from pathlib import Path
logger =  logging.getLogger(__name__)
class DataExtraction:
    
    def __init__(self,channel_url='https://www.youtube.com/@swc1611',output_file_name='youtube_videos'):
        self.channel_url = channel_url
        self.output_file_name = output_file_name
        
        
    def extract_videos_metadata(self,output_file_name='swc_videos_metadata'):
        self.output_file_name = f"{output_file_name}.json"
            
        flat_opts = {
        "extract_flat": True,
        "quiet": True,
            }

        videos = []

        with yt_dlp.YoutubeDL(flat_opts) as ydl:
            info = ydl.extract_info(self.channel_url, download=False)

            for data in info['entries']:
                for entry in data["entries"]:

                    video_url = f"https://www.youtube.com/watch?v={entry['id']}"

                    videos.append(video_url)

        # Second pass: full metadata
        full_opts = {
            "quiet": True,
             
        }

        cleaned = []

        with yt_dlp.YoutubeDL(full_opts) as ydl:

            for video_url in videos:

                try:
                    video_info = ydl.extract_info(
                        video_url,
                        download=False,
                        
                    )
                    cleaned.append({
                        "video_title": video_info.get("title"),
                        "video_id": video_info.get("id"),
                        "video_url": video_url,

                        "description": video_info.get("description"),

                        "duration": video_info.get("duration"),

                        "view_count": video_info.get("view_count"),

                        "like_count": video_info.get("like_count"),

                        "channel": video_info.get("channel"),

                        "channel_id": video_info.get("channel_id"),

                        "upload_date": video_info.get("upload_date"),

                        "tags": video_info.get("tags"),

                        "categories": video_info.get("categories"),

                        "thumbnail": video_info.get("thumbnail"),

                        "playlist": video_info.get("playlist"),

                        "chapters": video_info.get("chapters"),

                        "subtitles": list(video_info.get("subtitles", {}).keys())
                    })

                except Exception as e:
                    print(f"Error with {video_url}: {e}")

        with open(self.output_file_name, "w", encoding="utf-8") as f:
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
    
    def __merge_transcript_chunks(self, transcript_data,metadata,chunk_size=5):
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
                "start": int(chunk_group[0]["start"]),   # start of first object
                "end": int(chunk_group[-1]["end"]),       # end of last object
                "video_title" : metadata['video_title'],
                "tags" : metadata['tags'],
                "video_id":metadata['video_id']
            }

            merged_data.append(merged_object)

        return merged_data
    
    def read_json(self,input_dir): 

        with open(input_dir,'r',encoding='utf-8') as f : 
            data = json.load(f)
        return data 
    def change_chunk_metadata(self,subs_dir='swc_subs'):
        
        '''
            this function was created to handle the repeated chunks, 
            and change the 
        '''
        
        for file in os.listdir(subs_dir):
            full_path = os.path.join(os.getcwd(),subs_dir,file)
            data = self.read_json(full_path)

            # Handle empty file
            if not data:
                raise ValueError("JSON file is empty")

            first_item = data[0]

            video_metadata = {
                "video_title": first_item.get("video_title"),
                "video_id": first_item.get("video_id"),
                "video_url": first_item.get("video_url"),
                "description": first_item.get("description"),
                "duration": first_item.get("duration"),
                "view_count": first_item.get("view_count"),
                "like_count": first_item.get("like_count"),
                "channel": first_item.get("channel"),
                "channel_id": first_item.get("channel_id"),
                "upload_date": first_item.get("upload_date"),
                "tags": first_item.get("tags"),
                "categories": first_item.get("categories"),
                "thumbnail": first_item.get("thumbnail"),
                "playlist": first_item.get("playlist"),
                "chapters": first_item.get("chapters"),
                "subtitles": first_item.get("subtitles")
            }
            

            chunks = []

            for item in data:
                chunks.append({
                    "chunk_id": item.get("chunk_id"),
                    "text": item.get("text"),
                    "start": item.get("start"),
                    "end": item.get("end")
                })


            new_schema = {
                "video_metadata": video_metadata,
                "chunks": chunks
            }

            # Save new schema
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(
                    new_schema,
                    f,
                    ensure_ascii=False,
                    indent=4
                )
        
        print("done")
        
    def add_meta_data_to_chunks(self,subs_dir='./swc_subs'):
        file_path = Path(__file__).resolve().parent / '..' / 'swc_videos_metadata.json'
        print(file_path)
        videos_data = self.read_json(file_path)
        for file in os.listdir(subs_dir):
            video_id = file.replace("transcribe_", "").replace(".json", "")
            wanted_metadata = None
            for metadata in videos_data : 
                if metadata['video_id'] == video_id:
                    wanted_metadata = metadata
                    break
            full_path = os.path.join(os.getcwd(),subs_dir,file) 
            chunks = self.read_json(full_path)
            for chunk in chunks : 
                chunk.update(wanted_metadata)
            with open(full_path,'w',encoding='utf-8') as f: 
                json.dump(chunks,f,ensure_ascii=False, indent=4)
    
    def make_merge(self,input_dir='swc_subs',output_dir='merged_subs',chunk_size=5):
        output_dir = os.path.join(os.getcwd(),output_dir)
        os.makedirs(output_dir,exist_ok=True)
        for file in os.listdir(input_dir):
            full_file_path = os.path.join(input_dir,file)
            data = self.read_json(full_file_path)
            merged = self.__merge_transcript_chunks(
                data['chunks'],
                metadata=data["video_metadata"],
                chunk_size=chunk_size)
            new_file = os.path.join(output_dir,file)
            with open(f"{new_file}",'w',encoding='utf-8') as f : 
                json.dump(merged,f,ensure_ascii=False, indent=4)
            logger.info(f"Done with file {full_file_path}.")
        print('Done')
        
    
    def get_video_metadata_with_playlist(self, input_file=None, output_file=None):
        if input_file is None:
            input_file = Path(__file__).resolve().parent.parent / 'swc_videos_metadata.json'

        videos = self.read_json(input_file)
        for video in videos:
            if not video.get("playlist"):
                video["playlist"] = self._infer_playlist_from_metadata(video)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(videos, f, ensure_ascii=False, indent=4)

        return videos

    def extract_videos_metadata_playlist(self, output_file_name='swc_videos_metadata_playlist'):
        output_file = f"{output_file_name}.json"
        videos = self.get_video_metadata_with_playlist(output_file=output_file)
        return videos
    


    def get_playlist_metadata(
        self,
        playlist_title,
        playlist_url
        ):
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'extract_flat': True,    # CRITICAL: This makes it FAST by not grabbing full details for every video
        }
        
        playlist_videos = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_dict = ydl.extract_info(playlist_url, download=False)
            
            # Access the list of videos
            videos = playlist_dict.get('entries', [])
            for video in videos : 
                new_video_metadata = {}
                new_video_metadata['playlist_title'] = playlist_title
                new_video_metadata['video_id'] = video['id']
                new_video_metadata['video_url'] = video['url']
                playlist_videos.append(new_video_metadata)
                
        return playlist_videos
                
            
             
            
    def get_channel_playlists(self):
        # Ensure the URL points to the playlists tab
        channel_url = self.channel_url
        if not channel_url.endswith('/playlists'):
            channel_url = channel_url.rstrip('/') + '/playlists'
        
        ydl_opts = {
            'extract_flat': True,  # Only fetch metadata, don't download videos
            'quiet': True,
        }
        videos_metadata_playlist = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract channel info
            info = ydl.extract_info(channel_url, download=False)
            
            # 'entries' contains the list of playlists
            playlists = info.get('entries', [])
            
            for playlist in playlists:
                
                
                playlist_title = playlist['title']
                playlist_url = playlist['url']
                videos_metadata_playlist.append( 
                                                self.get_playlist_metadata(playlist_url=playlist_url,playlist_title=playlist_title)
                                                )
        with open('videos_playlist_title.json', "w", encoding="utf-8") as f:
            json.dump(videos_metadata_playlist, f, ensure_ascii=False, indent=4)

                
    def add_playlist(self,playlist_json_file,videos_file):
        playlist_data = self.read_json(playlist_json_file)
        videos_data = self.read_json(videos_file)
        ids_2idx = {}
        
        for playlist in playlist_data : 
            for video_playlist in playlist: 
                if video_playlist['video_id'] not in ids_2idx: 
                    for idx , video in enumerate(videos_data) :
                        if video['video_id'] == video_playlist['video_id'] : 
                            ids_2idx['video_id'] = idx 
                            if video['playlist'] is None : 
                                video['playlist']=[video_playlist['playlist_title']]
                                
                            else : 
                                video['playlist'].append(video_playlist['playlist_title'])                 
                else : 
                    idx = ids_2idx[video_playlist['video_id']]
                    videos_data[idx]['playlist'].append(video_playlist['playlist_title'])
                    
                                
        with open('video_metadata_v1.json','w') as f:
            json.dump(videos_data,f,ensure_ascii=False, indent=4)
        
        
    def get_all_playlist_names(self,file_name = 'video_metadata_v1'):
        file_name = f'{file_name}.json'
        unique_playlist = set()
        data = self.read_json(file_name)
        for video in data : 
            if video['playlist']:
                for playlist in video['playlist'] : 
                    if playlist not in unique_playlist : 
                        unique_playlist.add(playlist)
        
        return list(unique_playlist)
        
        
    def add_playlist_to_chunks(self, chunks_dir='merged_subs', videos_metadata_input='video_metadata_v1.json'):
        # Load videos metadata once (can be file path or already loaded list)
        if isinstance(videos_metadata_input, str):
            videos_metadata = self.read_json(videos_metadata_input)
        else:
            videos_metadata = videos_metadata_input
        
        for file in tqdm(os.listdir(chunks_dir)):
            video_id = file.replace("transcribe_", "").replace(".json", "")
            chunk_file_path = os.path.join(os.getcwd(), chunks_dir, file)
            chunks = self.read_json(chunk_file_path)
            
            playlist_name = None
            for video in videos_metadata:
                if video['video_id'] == video_id:
                    playlist_name = video.get('playlist')
                    break
            
            for chunk in chunks:
                chunk['playlist'] = playlist_name
            
            os.makedirs('merged_subs_2', exist_ok=True)
            new_file_path = os.path.join(os.getcwd(), 'merged_subs_2', file)
            with open(new_file_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
            
        
        
if __name__ == '__main__':
    extractor = DataExtraction()
    
    playlist_names = extractor.get_all_playlist_names()
    with open('playlist_names.json','w',encoding='utf-8') as f : 
        json.dump(playlist_names,f,ensure_ascii=False,indent=4)
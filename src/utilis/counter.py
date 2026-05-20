

import json 


with open('./youtube_videos.json','r',encoding='utf-8') as f : 
    data = json.load(f)
    
counter = 0
start = False  
for video in data : 
    if video['video_id'] == 'Hp4Q1VGA57Y':
        start = True
    if not start : continue 
    if video['video_duration'] is not None :
        counter+=video['video_duration'] 
    
    
print(int(counter / (3600)))
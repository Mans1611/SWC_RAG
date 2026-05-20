import json 
import os 
from src.DataExtraction import DataExtraction
extractor = DataExtraction()


def word_analytics(dir_path,output_name='words_count'):
    words_count = {}
    files_list = os.listdir(dir_path)
    for file in files_list : 
        full_file_path = os.path.join(dir_path,file)
        data = extractor.read_json(full_file_path)
        for chunk  in data :
            for text in chunk['text'].split(' ') : 
                if text not in words_count : 
                    words_count[text] = 0 
                    
                words_count[text]+=1
    
    words_count = dict(sorted(words_count.items(), key=lambda x: x[1], reverse=True))
    with open(f'{output_name}.json','w') as f :
        json.dump(words_count,f,ensure_ascii=False,indent=4)
        
word_analytics('merged_subs')
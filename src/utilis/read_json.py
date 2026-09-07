import json

def read_json(input_dir): 
    print(input_dir)
    with open(input_dir,'r',encoding='utf-8') as f : 
        data = json.load(f)
    return data 

def write_josn(full_path:str,data):
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )
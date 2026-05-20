from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Chunking:
    def __init__(self,chunk_size=1000,overlap=200):
        self.chunk_size = chunk_size 
        self.overlap = overlap
        
    def create_chunks(self, transcript_data):

        full_text = ""
        timestamps = []

        for item in transcript_data:

            start_char = len(full_text)

            full_text += item["text"] + " "

            end_char = len(full_text)

            timestamps.append({
                "start_char": start_char,
                "end_char": end_char,
                "start_time": int(item["start"]),
                "end_time": int(item["end"])
            })
        video_title = transcript_data[0]['video_title']

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=[
                "\n\n",
                "\n",
                ".",
                "؟",
                "!",
                " ",
                ""
            ]
        )

        chunks = splitter.split_text(full_text)

        documents = []

        current_position = 0

        for idx, chunk in enumerate(chunks):

            chunk_start = full_text.find(chunk, current_position)

            chunk_end = chunk_start + len(chunk)

            current_position = chunk_end

            start_time = None
            end_time = None

            for ts in timestamps:

                if ts["end_char"] >= chunk_start and start_time is None:
                    start_time = ts["start_time"]

                if ts["start_char"] <= chunk_end:
                    end_time = ts["end_time"]
            print('****'*30)
            print(chunk)
            print('****'*30)
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "chunk_id": idx,
                        "start": start_time,
                        "end": end_time,
                        "video_title":video_title
                    }
                )
            )

        return documents
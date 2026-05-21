import pandas as pd
import os
from pathlib import Path
import json

class RAGEvaluator:
    def __init__(self, csv_file_path: str = "rag_output.csv"):
        """
        Initialize the RAG Evaluator.
        
        Args:
            csv_file_path: Path to the CSV file where results will be saved
        """
        self.csv_file_path = csv_file_path
        self.columns = ['user_question', 'llm_response', 'video_id', 'true_answer', 'start', 'end']
        
        # Create DataFrame with columns if CSV doesn't exist
        if not os.path.exists(self.csv_file_path):
            self.df : pd.DataFrame = pd.DataFrame(columns=self.columns)
            self.df.to_csv(csv_file_path)
    
    def load_csv(self) -> pd.DataFrame:
        """Load existing CSV file or create empty DataFrame."""
        if os.path.exists(self.csv_file_path):
            return pd.read_csv(self.csv_file_path)
        return pd.DataFrame(columns=self.columns)
    
    def add_record(self, 
                   user_question: str, 
                   llm_response: str, 
                   video_id: str, 
                   start: float, 
                   end: float, 
                   true_answer: str = "") -> None:
        """
        Add a new record to the dataframe and save to CSV.
        
        Args:
            user_question: The user's question
            llm_response: The LLM's response
            video_id: The video ID from retrieved context
            start: Start timestamp
            end: End timestamp
            true_answer: Optional true answer for evaluation (can be filled later)
        """
        # Load existing data
        df = self.load_csv()
        
        # Create new record
        new_record = {
            'user_question': user_question,
            'llm_response': llm_response,
            'video_id': video_id,
            'true_answer': true_answer,
            'start': start,
            'end': end
        }
        
        # Append to dataframe
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        
        # Save to CSV
        df.to_csv(self.csv_file_path, index=False)
        
        print(f"✓ Record saved to {self.csv_file_path}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the current dataframe."""
        return self.load_csv()
    
    def get_stats(self) -> dict:
        """Get evaluation statistics."""
        df = self.load_csv()
        return {
            'total_records': len(df),
            'unique_videos': df['video_id'].nunique() if len(df) > 0 else 0,
            'records_with_true_answer': len(df[df['true_answer'].notna()]) if len(df) > 0 else 0
        }


# Global evaluator instance
rag_evaluator = RAGEvaluator()

from fastapi import APIRouter
from ..utils.rag_evaluator import rag_evaluator
import pandas as pd

evaluator_router = APIRouter()

@evaluator_router.get('/stats/')
async def get_stats():
    """Get evaluation statistics."""
    return rag_evaluator.get_stats()

@evaluator_router.get('/data/')
async def get_evaluation_data():
    """Get all evaluation data as JSON."""
    df = rag_evaluator.get_dataframe()
    return df.to_dict('records')

@evaluator_router.put('/update/{record_index}')
async def update_true_answer(record_index: int, true_answer: str):
    """
    Update the true_answer field for a specific record.
    
    Args:
        record_index: The index of the record to update
        true_answer: The true answer to set
    """
    try:
        df = rag_evaluator.load_csv()
        if record_index < 0 or record_index >= len(df):
            return {"error": "Invalid record index"}
        
        df.loc[record_index, 'true_answer'] = true_answer
        df.to_csv(rag_evaluator.csv_file_path, index=False)
        
        return {
            "status": "success",
            "message": f"Updated record {record_index}",
            "record": df.iloc[record_index].to_dict()
        }
    except Exception as e:
        return {"error": str(e)}

@evaluator_router.get('/export/')
async def export_data():
    """Get evaluation data summary."""
    df = rag_evaluator.get_dataframe()
    if len(df) == 0:
        return {
            "message": "No evaluation data yet",
            "file": rag_evaluator.csv_file_path
        }
    
    return {
        "total_records": len(df),
        "file": rag_evaluator.csv_file_path,
        "columns": list(df.columns),
        "preview": df.head(10).to_dict('records')
    }

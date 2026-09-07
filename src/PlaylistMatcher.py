from sentence_transformers import SentenceTransformer
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class PlaylistMatcher:
    """
    Matches user queries to playlists using embedding similarity.
    Uses SentenceTransformer to embed questions and playlist information,
    then finds the most similar playlist based on cosine similarity.
    """

    def __init__(self, playlist_tags_file: str = None, embedding_model: str = "BAAI/bge-m3"):
        """
        Initialize the PlaylistMatcher.
        
        Args:
            playlist_tags_file: Path to the playlist_tags.json file
            embedding_model: Model to use for embeddings (default: BAAI/bge-m3)
        """
        if playlist_tags_file is None:
            # Look for playlist_tags.json in the parent directory
            playlist_tags_file = Path(__file__).resolve().parent.parent / 'playlist_tags.json'
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.playlist_tags_file = playlist_tags_file
        self.playlists = self._load_playlists()
        self.playlist_embeddings = self._create_playlist_embeddings()

    def _load_playlists(self) -> List[Dict]:
        """Load playlists from JSON file."""
        with open(self.playlist_tags_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_playlist_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Create embeddings for each playlist by combining the name and tags.
        Returns a dictionary mapping playlist names to their embeddings.
        """
        playlist_embeddings = {}
        
        for playlist in self.playlists:
            playlist_name = playlist['playlist_name']
            tags = playlist['tags']
            
            # Combine playlist name and tags into a single text
            combined_text = f"{playlist_name} {' '.join(tags)}"
            
            # Get embedding
            embedding = self.embedding_model.encode(
                combined_text,
                normalize_embeddings=True
            )
            
            playlist_embeddings[playlist_name] = embedding
        
        return playlist_embeddings

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(vec1, vec2))

    def find_matching_playlist(
        self,
        question: str,
        top_k: int = 3,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Find the most similar playlist(s) for a user question.
        
        Args:
            question: User's question/query in English or Arabic
            top_k: Number of top matches to return (default: 3)
            threshold: Minimum similarity score to include (default: 0.0)
        
        Returns:
            List of dictionaries with playlist info and similarity scores,
            sorted by similarity score (highest first).
            
        Example:
            >>> matcher = PlaylistMatcher()
            >>> results = matcher.find_matching_playlist("شرح قاعدة السلسلة")
            >>> for result in results:
            ...     print(f"{result['playlist_name']}: {result['similarity_score']:.4f}")
        """
        # Embed the question
        question_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True
        )
        
        # Calculate similarity scores for all playlists
        results = []
        for playlist in self.playlists:
            playlist_name = playlist['playlist_name']
            playlist_embedding = self.playlist_embeddings[playlist_name]
            
            similarity_score = self._cosine_similarity(
                question_embedding,
                playlist_embedding
            )
            
            if similarity_score >= threshold:
                results.append({
                    'playlist_name': playlist_name,
                    'similarity_score': similarity_score,
                    'tags': playlist['tags']
                })
        
        # Sort by similarity score (descending)
        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top_k results
        return results[:top_k]

    def find_best_playlist(self, question: str) -> Dict:
        """
        Find the single best matching playlist for a question.
        
        Args:
            question: User's question/query
        
        Returns:
            Dictionary with the best matching playlist and its similarity score
        """
        results = self.find_matching_playlist(question, top_k=1)
        return results[0] if results else None

    def get_all_playlists(self) -> List[Dict]:
        """Get all available playlists with their tags."""
        return self.playlists

    def get_playlist_tags(self, playlist_name: str) -> List[str]:
        """Get tags for a specific playlist."""
        for playlist in self.playlists:
            if playlist['playlist_name'] == playlist_name:
                return playlist['tags']
        return None

    def batch_find_matching_playlists(
        self,
        questions: List[str],
        top_k: int = 3
    ) -> List[List[Dict]]:
        """
        Find matching playlists for multiple questions.
        
        Args:
            questions: List of user questions
            top_k: Number of top matches per question
        
        Returns:
            List of results for each question
        """
        return [self.find_matching_playlist(q, top_k=top_k) for q in questions]


if __name__ == '__main__':
    # Example usage
    matcher = PlaylistMatcher()
    
    # Test queries in both English and Arabic
    test_queries = [
        "تكامل بالأجزاء",
        "integration by parts",
        "شرح المصفوفات",
        "linear transformations",
        "حساب الحجم باستخدام التكامل",
        "differential equations applications",
        "معادلات من الرتبة الثانية",
        "hyperbolic functions",
        "حل أنظمة معادلات خطية",
        "limit calculation"
    ]
    
    print("=" * 80)
    print("PLAYLIST MATCHER RESULTS")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        results = matcher.find_matching_playlist(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['playlist_name']}")
            print(f"   Similarity Score: {result['similarity_score']:.4f}")
            print(f"   Top Tags: {', '.join(result['tags'][:4])}")

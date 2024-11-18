from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from typing import List, Union
import numpy as np
import os
import time

class EmbeddingModel:
    def __init__(self, dimensions: int = 1024):
        """Initialize the Mixedbread embedding model using local files."""
        self.dimensions = dimensions
        self.model = None
        self._model_path = None
        
        # Get the absolute path to the model directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self._model_path = os.path.join(current_dir, 'mxbai-embed-large-v1')
        
        if not os.path.exists(self._model_path):
            raise ValueError(
                f"Model not found at {self._model_path}. "
                "Please ensure the model files are in the correct location."
            )

    def _ensure_model_loaded(self):
        """Ensure the model is loaded before use."""
        if self.model is None:
            print("Loading embedding model... This may take a moment.")
            start_time = time.time()
            try:
                self.model = SentenceTransformer(self._model_path)
                print(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query with the appropriate retrieval prompt."""
        try:
            self._ensure_model_loaded()
            
            # Add the required prompt for queries
            query_prompt = "Represent this sentence for searching relevant passages: "
            embeddings = self.model.encode(f"{query_prompt}{query}", normalize_embeddings=True)
            
            # Ensure proper shape
            if isinstance(embeddings, np.ndarray):
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.reshape(1, -1)
            
            return embeddings
        except Exception as e:
            print(f"Error encoding query: {str(e)}")
            raise
    
    def encode_documents(self, documents: Union[str, List[str]]) -> np.ndarray:
        """Encode documents without the query prompt."""
        try:
            self._ensure_model_loaded()
            
            embeddings = self.model.encode(documents, normalize_embeddings=True)
            
            # Ensure proper shape
            if isinstance(embeddings, np.ndarray):
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.reshape(1, -1)
            
            return embeddings
        except Exception as e:
            print(f"Error encoding documents: {str(e)}")
            raise
    
    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document embeddings."""
        try:
            # Ensure proper shapes for similarity computation
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            if len(doc_embeddings.shape) == 1:
                doc_embeddings = doc_embeddings.reshape(1, -1)
                
            return cos_sim(query_embedding, doc_embeddings)
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            raise
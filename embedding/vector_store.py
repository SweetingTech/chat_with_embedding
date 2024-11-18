import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import hashlib
from .model import EmbeddingModel

class VectorStore:
    def __init__(self, persist_directory: str = "vector_db"):
        """Initialize the vector store with ChromaDB and embedding model."""
        try:
            # Initialize embedding model
            print("Initializing embedding model...")
            self.embedding_model = EmbeddingModel()
            
            # Initialize ChromaDB
            print("Initializing ChromaDB...")
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Create or get the collection for documents
            self.collection = self.client.get_or_create_collection(
                name="document_store",
                metadata={"hnsw:space": "cosine"}
            )
            print("Vector store initialization complete.")
        except Exception as e:
            print(f"Error initializing VectorStore: {str(e)}")
            raise

    def _get_document_hash(self, content: str) -> str:
        """Generate a unique hash for document content."""
        return hashlib.md5(content.encode()).hexdigest()

    def add_document(self, filename: str, content: str) -> None:
        """Add or update a document in the vector store."""
        try:
            # Split content into chunks (simple sentence-based splitting for now)
            chunks = [s.strip() for s in content.split('.') if s.strip()]
            
            if not chunks:
                print(f"Warning: No valid chunks found in document {filename}")
                return
            
            print(f"Processing document '{filename}' into {len(chunks)} chunks...")
            
            # Generate document hash
            doc_hash = self._get_document_hash(content)
            
            # Generate unique IDs for each chunk
            chunk_ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]
            
            # Create metadata for each chunk
            metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
            
            # Generate embeddings for chunks
            print("Generating embeddings...")
            embeddings = self.embedding_model.encode_documents(chunks)
            
            # Convert embeddings to list format
            embeddings_list = embeddings.tolist()
            
            # Add chunks to ChromaDB
            print("Adding to vector store...")
            self.collection.add(
                embeddings=embeddings_list,
                documents=chunks,
                ids=chunk_ids,
                metadatas=metadatas
            )
            print(f"Successfully added document '{filename}' to vector store")
            
        except Exception as e:
            print(f"Error adding document to vector store: {str(e)}")
            raise

    def remove_document(self, filename: str) -> None:
        """Remove a document from the vector store."""
        try:
            results = self.collection.get(
                where={"filename": filename}
            )
            if results and results['ids']:
                self.collection.delete(
                    ids=results['ids']
                )
                print(f"Successfully removed document '{filename}' from vector store")
            else:
                print(f"Document '{filename}' not found in vector store")
        except Exception as e:
            print(f"Error removing document from vector store: {str(e)}")
            raise

    def query_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the vector store for similar content."""
        try:
            if not query.strip():
                print("Warning: Empty query received")
                return []
                
            print(f"Generating embedding for query: {query}")
            # Generate query embedding
            query_embedding = self.embedding_model.encode_query(query)
            
            print("Querying vector store...")
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            
            if not results or not results['documents']:
                print(f"No matching results found for query: {query}")
                return []
            
            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    'content': doc,
                    'filename': metadata['filename'],
                    'chunk_index': metadata['chunk_index'],
                    'similarity': 1 - distance  # Convert distance to similarity score
                })
            
            print(f"Found {len(formatted_results)} matching results")
            return formatted_results
            
        except Exception as e:
            print(f"Error querying vector store: {str(e)}")
            raise
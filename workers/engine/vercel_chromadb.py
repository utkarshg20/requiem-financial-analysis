"""
Vercel-compatible ChromaDB wrapper that uses in-memory storage
for serverless deployment
"""
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import os

logger = logging.getLogger(__name__)

class VercelChromaDB:
    """
    ChromaDB wrapper optimized for Vercel serverless deployment
    Uses in-memory storage instead of persistent SQLite
    """
    
    def __init__(self):
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client for Vercel"""
        try:
            # Use in-memory storage for Vercel
            self.client = chromadb.Client(Settings(
                persist_directory=None,  # No persistence for serverless
                anonymized_telemetry=False
            ))
            
            # Create or get collection
            collection_name = "earnings_documents"
            try:
                self.collection = self.client.get_collection(collection_name)
            except:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Earnings call documents"}
                )
            
            logger.info("Vercel ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Vercel ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any], embedding: List[float]):
        """Add document to collection"""
        if not self.collection:
            logger.error("ChromaDB collection not initialized")
            return False
        
        try:
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                embeddings=[embedding],
                ids=[doc_id]
            )
            logger.info(f"Document {doc_id} added to Vercel ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error adding document to Vercel ChromaDB: {e}")
            return False
    
    def search_documents(self, query_embedding: List[float], ticker: str = None, quarter: str = None, limit: int = 5):
        """Search for similar documents"""
        if not self.collection:
            logger.error("ChromaDB collection not initialized")
            return []
        
        try:
            where_clause = {}
            if ticker:
                where_clause["ticker"] = ticker
            if quarter:
                where_clause["quarter"] = quarter
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause if where_clause else None
            )
            
            # Convert to our expected format
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    documents.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0
                    })
            
            logger.info(f"Found {len(documents)} documents in Vercel ChromaDB")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching Vercel ChromaDB: {e}")
            return []
    
    def get_collection_info(self):
        """Get collection information"""
        if not self.collection:
            return {"count": 0, "name": "not_initialized"}
        
        try:
            count = self.collection.count()
            return {"count": count, "name": self.collection.name}
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"count": 0, "name": "error"}

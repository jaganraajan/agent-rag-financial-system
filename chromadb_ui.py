#!/usr/bin/env python3
"""
ChromaDB Inspection UI

A simple web interface to inspect ChromaDB contents and chunking quality.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, request, jsonify, send_from_directory
import chromadb
from chromadb.config import Settings
from src.rag.rag_pipeline import VectorStore, EmbeddingService
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ChromaDBInspector:
    """Inspector for ChromaDB contents."""
    
    def __init__(self, vector_store_path: str = "./vector_db"):
        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.embedding_service = EmbeddingService()
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize connection to ChromaDB."""
        try:
            self.vector_store = VectorStore(self.vector_store_path)
            logger.info(f"Connected to ChromaDB at {self.vector_store_path}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            self.vector_store = None
    
    def get_collection_stats(self) -> Dict:
        """Get basic statistics about the collection."""
        if not self.vector_store:
            return {"error": "Not connected to ChromaDB"}
        
        try:
            info = self.vector_store.get_collection_info()
            
            # Get additional stats by querying all chunks
            all_chunks = self.get_all_chunks(limit=None)
            
            # Calculate statistics
            section_types = {}
            content_types = {}
            token_counts = []
            
            for chunk in all_chunks:
                metadata = chunk.get('metadata', {})
                section_type = metadata.get('section_type', 'unknown')
                content_type = metadata.get('content_type', 'unknown')
                token_count = metadata.get('token_count', 0)
                
                section_types[section_type] = section_types.get(section_type, 0) + 1
                content_types[content_type] = content_types.get(content_type, 0) + 1
                if token_count > 0:
                    token_counts.append(token_count)
            
            # Calculate token statistics
            token_stats = {}
            if token_counts:
                token_stats = {
                    'min': min(token_counts),
                    'max': max(token_counts),
                    'avg': sum(token_counts) / len(token_counts),
                    'total': sum(token_counts)
                }
            
            info.update({
                'section_types': section_types,
                'content_types': content_types,
                'token_stats': token_stats
            })
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def get_all_chunks(self, limit: Optional[int] = 100) -> List[Dict]:
        """Get all chunks from the collection."""
        if not self.vector_store:
            return []
        
        try:
            # ChromaDB doesn't have a direct "get all" method, so we use a dummy query
            # to get all documents with their metadata
            results = self.vector_store.collection.get(
                include=['documents', 'metadatas', 'embeddings'],
                limit=limit
            )
            
            chunks = []
            for i in range(len(results['documents'])):
                chunk = {
                    'id': results['ids'][i] if 'ids' in results else f"chunk_{i}",
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i] if results['metadatas'] else {},
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting all chunks: {e}")
            return []
    
    def search_chunks(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for chunks similar to the query."""
        if not self.vector_store:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embedding_service.get_embedding(query)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, top_k)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            return []
    
    def get_chunks_by_section(self, section_type: str) -> List[Dict]:
        """Get all chunks from a specific section type."""
        all_chunks = self.get_all_chunks(limit=None)
        return [chunk for chunk in all_chunks 
                if chunk.get('metadata', {}).get('section_type') == section_type]

# Initialize inspector
inspector = ChromaDBInspector()

@app.route('/')
def index():
    """Main dashboard page."""
    stats = inspector.get_collection_stats()
    return render_template('index.html', stats=stats)

@app.route('/api/stats')
def api_stats():
    """API endpoint for collection statistics."""
    return jsonify(inspector.get_collection_stats())

@app.route('/api/chunks')
def api_chunks():
    """API endpoint to get all chunks."""
    limit = request.args.get('limit', 100, type=int)
    section_type = request.args.get('section_type', None)
    
    if section_type:
        chunks = inspector.get_chunks_by_section(section_type)
    else:
        chunks = inspector.get_all_chunks(limit)
    
    return jsonify(chunks)

@app.route('/api/search')
def api_search():
    """API endpoint for chunk search."""
    query = request.args.get('query', '')
    top_k = request.args.get('top_k', 10, type=int)
    
    if not query:
        return jsonify([])
    
    results = inspector.search_chunks(query, top_k)
    return jsonify(results)

@app.route('/chunks')
def chunks_view():
    """Page to view and browse chunks."""
    return render_template('chunks.html')

@app.route('/search')
def search_view():
    """Page for semantic search."""
    return render_template('search.html')

# Create templates directory if it doesn't exist
templates_dir = Path(__file__).parent / 'templates'
templates_dir.mkdir(exist_ok=True)

if __name__ == '__main__':
    port = 8080  # Use a simpler port
    app.run(debug=False, host='0.0.0.0', port=port)  # Disable debug to avoid reloader issues
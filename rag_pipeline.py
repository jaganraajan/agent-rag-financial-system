#!/usr/bin/env python3
"""
RAG Pipeline for Financial Documents

This module implements a Retrieval-Augmented Generation pipeline for SEC filings,
including text extraction, chunking, embedding, vector storage, and retrieval.
"""

import os
import re
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import tiktoken
import chromadb
from chromadb.config import Settings
from bs4 import BeautifulSoup
import openai
from openai import AzureOpenAI


class TextExtractor:
    """Extract and clean text from HTML SEC filings."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_html(self, file_path: str) -> str:
        """Extract clean text from HTML filing."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            text = self._clean_text(text)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


class TextChunker:
    """Split text into semantic chunks with token limits."""
    
    def __init__(self, min_tokens: int = 50, max_tokens: int = 1000, model: str = "cl100k_base"):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)
        
        # Try to initialize tiktoken, fallback to simple word-based counting
        try:
            self.encoding = tiktoken.get_encoding(model)
            self.use_tiktoken = True
            self.logger.info("Using tiktoken for token counting")
        except Exception as e:
            self.logger.warning(f"Failed to load tiktoken encoding: {e}")
            self.logger.info("Using word-based token estimation")
            self.encoding = None
            self.use_tiktoken = False
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.use_tiktoken and self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Simple word-based estimation (approximately 0.75 tokens per word)
            words = len(text.split())
            return int(words * 0.75)
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Split text into chunks with token limits."""
        chunks = []
        
        # Split by paragraphs first for semantic boundaries
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_tokens = self._count_tokens(paragraph)
            
            # If single paragraph exceeds max tokens, split by sentences
            if paragraph_tokens > self.max_tokens:
                sentences = self._split_by_sentences(paragraph)
                for sentence in sentences:
                    sentence_tokens = self._count_tokens(sentence)
                    
                    if current_tokens + sentence_tokens > self.max_tokens:
                        if current_tokens >= self.min_tokens:
                            chunks.append(self._create_chunk(current_chunk, metadata))
                            current_chunk = sentence
                            current_tokens = sentence_tokens
                        else:
                            current_chunk += " " + sentence
                            current_tokens += sentence_tokens
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                        current_tokens += sentence_tokens
            else:
                # Check if adding this paragraph exceeds max tokens
                if current_tokens + paragraph_tokens > self.max_tokens:
                    if current_tokens >= self.min_tokens:
                        chunks.append(self._create_chunk(current_chunk, metadata))
                        current_chunk = paragraph
                        current_tokens = paragraph_tokens
                    else:
                        current_chunk += "\n\n" + paragraph
                        current_tokens += paragraph_tokens
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                    current_tokens += paragraph_tokens
        
        # Add final chunk if it meets minimum requirements
        if current_chunk and current_tokens >= self.min_tokens:
            chunks.append(self._create_chunk(current_chunk, metadata))
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences."""
        # Simple sentence splitting by periods followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """Create a chunk dictionary with metadata."""
        chunk = {
            'text': text.strip(),
            'token_count': self._count_tokens(text)
        }
        
        if metadata:
            chunk.update(metadata)
        
        return chunk


class EmbeddingService:
    """Handle embeddings using Azure OpenAI (with mock fallback for testing)."""
    
    def __init__(self, azure_endpoint: Optional[str] = None, api_key: Optional[str] = None, 
                 api_version: str = "2024-02-01", model: str = "text-embedding-ada-002"):
        self.azure_endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        self.api_version = api_version
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # Initialize client if credentials are available
        self.client = None
        if self.azure_endpoint and self.api_key:
            try:
                self.client = AzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version
                )
                self.logger.info("Azure OpenAI client initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Azure OpenAI client: {e}")
                self.logger.info("Will use mock embeddings for testing")
        else:
            self.logger.info("No Azure OpenAI credentials provided, using mock embeddings")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if self.client:
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding
            except Exception as e:
                self.logger.error(f"Error getting embedding from Azure OpenAI: {e}")
                return self._mock_embedding(text)
        else:
            return self._mock_embedding(text)
    
    def _mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding for testing purposes."""
        # Simple hash-based mock embedding (1536 dimensions like text-embedding-ada-002)
        import hashlib
        
        # Create deterministic but varied embedding based on text content
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Generate 1536-dimensional vector
        embedding = []
        for i in range(1536):
            # Use hash bytes cyclically and apply some mathematical transformations
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Normalize to [-1, 1] range
            val = (byte_val - 127.5) / 127.5
            # Add some variation based on position
            val += 0.1 * (i % 10 - 5) / 10
            embedding.append(val)
        
        # Normalize the vector
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding


class VectorStore:
    """ChromaDB vector store for document chunks."""
    
    def __init__(self, persist_directory: str = "./vector_db"):
        self.persist_directory = persist_directory
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="financial_documents",
            metadata={"description": "SEC filing chunks with embeddings"}
        )
        
        self.logger.info(f"Vector store initialized at {persist_directory}")
    
    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Add chunks with embeddings to the vector store."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Prepare data for ChromaDB
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [{k: v for k, v in chunk.items() if k != 'text'} for chunk in chunks]
        ids = [f"chunk_{i}_{hash(chunk['text'])}" for i, chunk in enumerate(chunks)]
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        self.logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return search_results
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection.name
        }


class RAGPipeline:
    """Main RAG pipeline orchestrator."""
    
    def __init__(self, vector_store_path: str = "./vector_db"):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.text_extractor = TextExtractor()
        self.chunker = TextChunker()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(vector_store_path)
        
        self.logger.info("RAG Pipeline initialized")
    
    def process_directory(self, filings_dir: str) -> Dict:
        """Process all HTML files in a directory."""
        filings_path = Path(filings_dir)
        if not filings_path.exists():
            raise ValueError(f"Directory not found: {filings_dir}")
        
        html_files = list(filings_path.glob("*.htm")) + list(filings_path.glob("*.html"))
        
        if not html_files:
            raise ValueError(f"No HTML files found in {filings_dir}")
        
        total_chunks = 0
        processed_files = 0
        
        for file_path in html_files:
            try:
                self.logger.info(f"Processing {file_path.name}...")
                
                # Extract metadata from filename
                metadata = self._extract_metadata_from_filename(file_path.name)
                
                # Extract text
                text = self.text_extractor.extract_text_from_html(str(file_path))
                if not text:
                    self.logger.warning(f"No text extracted from {file_path.name}")
                    continue
                
                # Create chunks
                chunks = self.chunker.chunk_text(text, metadata)
                if not chunks:
                    self.logger.warning(f"No chunks created from {file_path.name}")
                    continue
                
                # Generate embeddings
                embeddings = []
                for chunk in chunks:
                    embedding = self.embedding_service.get_embedding(chunk['text'])
                    embeddings.append(embedding)
                
                # Add to vector store
                self.vector_store.add_chunks(chunks, embeddings)
                
                total_chunks += len(chunks)
                processed_files += 1
                
                self.logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {e}")
        
        result = {
            'processed_files': processed_files,
            'total_files': len(html_files),
            'total_chunks': total_chunks
        }
        
        self.logger.info(f"Processing complete: {result}")
        return result
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Query the RAG system."""
        try:
            # Get query embedding
            query_embedding = self.embedding_service.get_embedding(question)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, top_k)
            
            return {
                'question': question,
                'results': results,
                'top_k': top_k
            }
            
        except Exception as e:
            self.logger.error(f"Error during query: {e}")
            return {
                'question': question,
                'error': str(e),
                'results': []
            }
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        return self.vector_store.get_collection_info()
    
    def _extract_metadata_from_filename(self, filename: str) -> Dict:
        """Extract metadata from SEC filing filename."""
        # Expected format: COMPANY_10K_YEAR_ACCESSION.htm
        metadata = {'filename': filename}
        
        try:
            # Remove extension
            name_without_ext = filename.replace('.htm', '').replace('.html', '')
            parts = name_without_ext.split('_')
            
            if len(parts) >= 3:
                metadata['company'] = parts[0]
                metadata['form_type'] = parts[1]
                metadata['year'] = parts[2]
                
                if len(parts) >= 4:
                    metadata['accession_number'] = '_'.join(parts[3:])
        
        except Exception as e:
            self.logger.warning(f"Could not parse metadata from filename {filename}: {e}")
        
        return metadata
"""
Content extraction and chunking for document processing.
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional
import re
from pathlib import Path

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("sklearn and numpy not available. Install with: pip install scikit-learn numpy")

from .pdf_processor import ExtractedContent, BatchDocumentProcessor
from utils.config import config


logger = logging.getLogger(__name__)


class ContentChunker:
    """Intelligent content chunking for document processing."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
    
    def chunk_text(self, text: str, preserve_structure: bool = True) -> List[Dict[str, Any]]:
        """
        Chunk text into manageable pieces while preserving structure.
        
        Args:
            text: Text to chunk
            preserve_structure: Whether to preserve paragraph and sentence boundaries
            
        Returns:
            List of chunks with metadata
        """
        if not text:
            return []
        
        chunks = []
        
        if preserve_structure:
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            current_chunk = ""
            current_chunk_size = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                para_size = len(para)
                
                # If adding this paragraph exceeds chunk size
                if current_chunk_size + para_size > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        'text': current_chunk.strip(),
                        'start_pos': len(chunks) * self.chunk_size,
                        'end_pos': len(chunks) * self.chunk_size + len(current_chunk),
                        'chunk_id': len(chunks),
                        'word_count': len(current_chunk.split()),
                        'char_count': len(current_chunk)
                    })
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                        current_chunk = overlap_text + "\n\n" + para
                        current_chunk_size = len(overlap_text) + para_size + 2
                    else:
                        current_chunk = para
                        current_chunk_size = para_size
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + para
                        current_chunk_size += para_size + 2
                    else:
                        current_chunk = para
                        current_chunk_size = para_size
            
            # Add final chunk
            if current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_pos': len(chunks) * self.chunk_size,
                    'end_pos': len(chunks) * self.chunk_size + len(current_chunk),
                    'chunk_id': len(chunks),
                    'word_count': len(current_chunk.split()),
                    'char_count': len(current_chunk)
                })
        
        else:
            # Simple character-based chunking
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk_text = text[i:i + self.chunk_size]
                if chunk_text:
                    chunks.append({
                        'text': chunk_text,
                        'start_pos': i,
                        'end_pos': i + len(chunk_text),
                        'chunk_id': len(chunks),
                        'word_count': len(chunk_text.split()),
                        'char_count': len(chunk_text)
                    })
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= overlap_size:
            return text
        
        # Try to break at sentence boundary
        sentences = text.split('. ')
        if len(sentences) > 1:
            overlap_sentences = []
            current_size = 0
            
            for sentence in reversed(sentences):
                sentence_size = len(sentence) + 2  # Add 2 for '. '
                if current_size + sentence_size <= overlap_size:
                    overlap_sentences.insert(0, sentence)
                    current_size += sentence_size
                else:
                    break
            
            if overlap_sentences:
                return '. '.join(overlap_sentences) + ('.' if not overlap_sentences[-1].endswith('.') else '')
        
        # Fallback to character-based overlap
        return text[-overlap_size:]


class DocumentIndexer:
    """Create searchable indexes from extracted content."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.chunker = ContentChunker()
        self.documents = {}  # Store processed documents
        self.embeddings = {}  # Store document embeddings
    
    def index_document(self, file_path: str, content: ExtractedContent) -> str:
        """
        Index a document for searchability.
        
        Args:
            file_path: Path to the document file
            content: Extracted content from the document
            
        Returns:
            Document ID for referencing
        """
        doc_id = self._generate_doc_id(file_path)
        
        # Chunk the main text content
        text_chunks = self.chunker.chunk_text(content.text)
        
        # Process sections separately
        section_chunks = []
        if content.sections:
            for section in content.sections:
                section_text = f"{section.get('title', '')}\n{section.get('content', '')}"
                chunks = self.chunker.chunk_text(section_text)
                for chunk in chunks:
                    chunk['section_title'] = section.get('title', '')
                section_chunks.extend(chunks)
        
        # Store document data
        self.documents[doc_id] = {
            'file_path': file_path,
            'content': content,
            'text_chunks': text_chunks,
            'section_chunks': section_chunks,
            'title': content.title,
            'abstract': content.abstract,
            'authors': content.authors,
            'tables': content.tables,
            'figures': content.figures,
            'references': content.references,
            'equations': content.equations,
            'metadata': content.metadata
        }
        
        # Generate embeddings if LLM client is available
        if self.llm_client:
            self._generate_embeddings(doc_id)
        
        logger.info(f"Indexed document: {doc_id} with {len(text_chunks)} chunks")
        return doc_id
    
    def _generate_doc_id(self, file_path: str) -> str:
        """Generate a unique document ID."""
        file_name = os.path.basename(file_path)
        return re.sub(r'[^a-zA-Z0-9_-]', '_', file_name)
    
    def _generate_embeddings(self, doc_id: str) -> None:
        """Generate embeddings for document chunks."""
        try:
            doc = self.documents[doc_id]
            embeddings = {
                'text_chunks': [],
                'sections': [],
                'abstract': None,
                'title': None
            }
            
            # Generate embeddings for text chunks
            for chunk in doc['text_chunks']:
                try:
                    embedding = self.llm_client.generate_embeddings(chunk['text'])
                    embeddings['text_chunks'].append({
                        'chunk_id': chunk['chunk_id'],
                        'embedding': embedding,
                        'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text']
                    })
                except Exception as e:
                    logger.error(f"Error generating embedding for chunk {chunk['chunk_id']}: {e}")
            
            # Generate embeddings for sections
            for chunk in doc['section_chunks']:
                try:
                    embedding = self.llm_client.generate_embeddings(chunk['text'])
                    embeddings['sections'].append({
                        'chunk_id': chunk['chunk_id'],
                        'section_title': chunk.get('section_title', ''),
                        'embedding': embedding,
                        'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text']
                    })
                except Exception as e:
                    logger.error(f"Error generating embedding for section chunk: {e}")
            
            # Generate embeddings for abstract and title
            if doc['abstract']:
                try:
                    embeddings['abstract'] = self.llm_client.generate_embeddings(doc['abstract'])
                except Exception as e:
                    logger.error(f"Error generating embedding for abstract: {e}")
            
            if doc['title']:
                try:
                    embeddings['title'] = self.llm_client.generate_embeddings(doc['title'])
                except Exception as e:
                    logger.error(f"Error generating embedding for title: {e}")
            
            self.embeddings[doc_id] = embeddings
            logger.info(f"Generated embeddings for document: {doc_id}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings for {doc_id}: {e}")
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search indexed documents using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant document chunks with similarity scores
        """
        if not self.llm_client:
            logger.warning("No LLM client available for semantic search")
            return self._keyword_search(query, top_k)
        
        try:
            # Generate query embedding
            query_embedding = self.llm_client.generate_embeddings(query)
            results = []
            
            # Search through all documents
            for doc_id, doc_embeddings in self.embeddings.items():
                doc = self.documents[doc_id]
                
                # Search text chunks
                for chunk_emb in doc_embeddings.get('text_chunks', []):
                    similarity = self._calculate_similarity(query_embedding, chunk_emb['embedding'])
                    results.append({
                        'doc_id': doc_id,
                        'type': 'text_chunk',
                        'chunk_id': chunk_emb['chunk_id'],
                        'similarity': similarity,
                        'text': chunk_emb['text'],
                        'title': doc['title'],
                        'file_path': doc['file_path']
                    })
                
                # Search section chunks
                for chunk_emb in doc_embeddings.get('sections', []):
                    similarity = self._calculate_similarity(query_embedding, chunk_emb['embedding'])
                    results.append({
                        'doc_id': doc_id,
                        'type': 'section',
                        'chunk_id': chunk_emb['chunk_id'],
                        'section_title': chunk_emb['section_title'],
                        'similarity': similarity,
                        'text': chunk_emb['text'],
                        'title': doc['title'],
                        'file_path': doc['file_path']
                    })
                
                # Check abstract
                if doc_embeddings.get('abstract') and doc['abstract']:
                    similarity = self._calculate_similarity(query_embedding, doc_embeddings['abstract'])
                    results.append({
                        'doc_id': doc_id,
                        'type': 'abstract',
                        'similarity': similarity,
                        'text': doc['abstract'],
                        'title': doc['title'],
                        'file_path': doc['file_path']
                    })
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return self._keyword_search(query, top_k)
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            if not embedding1 or not embedding2:
                return 0.0
            
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback keyword-based search."""
        try:
            results = []
            query_lower = query.lower()
            
            for doc_id, doc in self.documents.items():
                # Search in text chunks
                for chunk in doc['text_chunks']:
                    if query_lower in chunk['text'].lower():
                        # Simple relevance scoring based on keyword frequency
                        relevance = chunk['text'].lower().count(query_lower) / len(chunk['text'].split())
                        results.append({
                            'doc_id': doc_id,
                            'type': 'text_chunk',
                            'chunk_id': chunk['chunk_id'],
                            'similarity': relevance,
                            'text': chunk['text'][:500] + '...' if len(chunk['text']) > 500 else chunk['text'],
                            'title': doc['title'],
                            'file_path': doc['file_path']
                        })
                
                # Search in abstract
                if doc['abstract'] and query_lower in doc['abstract'].lower():
                    relevance = doc['abstract'].lower().count(query_lower) / len(doc['abstract'].split())
                    results.append({
                        'doc_id': doc_id,
                        'type': 'abstract',
                        'similarity': relevance,
                        'text': doc['abstract'],
                        'title': doc['title'],
                        'file_path': doc['file_path']
                    })
            
            # Sort by relevance
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        return self.documents.get(doc_id)
    
    def list_documents(self) -> List[str]:
        """List all indexed document IDs."""
        return list(self.documents.keys())
    
    def save_index(self, index_path: str) -> None:
        """Save the document index to disk."""
        try:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # Prepare data for serialization (excluding embeddings due to size)
            index_data = {
                'documents': {},
                'config': {
                    'chunk_size': self.chunker.chunk_size,
                    'chunk_overlap': self.chunker.chunk_overlap
                }
            }
            
            for doc_id, doc in self.documents.items():
                # Convert ExtractedContent to dict
                content = doc['content']
                content_dict = {
                    'text': content.text,
                    'title': content.title,
                    'authors': content.authors or [],
                    'abstract': content.abstract,
                    'sections': content.sections or [],
                    'tables': content.tables or [],
                    'figures': content.figures or [],
                    'references': content.references or [],
                    'equations': content.equations or [],
                    'metadata': content.metadata or {}
                }
                
                index_data['documents'][doc_id] = {
                    'file_path': doc['file_path'],
                    'content': content_dict,
                    'text_chunks': doc['text_chunks'],
                    'section_chunks': doc['section_chunks'],
                    'title': doc['title'],
                    'abstract': doc['abstract'],
                    'authors': doc['authors'],
                    'tables': doc['tables'],
                    'figures': doc['figures'],
                    'references': doc['references'],
                    'equations': doc['equations'],
                    'metadata': doc['metadata']
                }
            
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved document index to: {index_path}")
            
        except Exception as e:
            logger.error(f"Error saving index to {index_path}: {e}")
            raise

"""
Core AI Document Q&A Agent implementation.
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

from processors.pdf_processor import BatchPDFProcessor, ExtractedContent
from processors.content_extractor import DocumentIndexer
from utils.llm_client import get_llm_client, BaseLLMClient
from utils.config import config


logger = logging.getLogger(__name__)


class DocumentQAAgent:
    """
    Enterprise-ready AI agent for document Q&A with multi-modal capabilities.
    """
    
    def __init__(self, llm_provider: str = None, documents_path: str = None):
        """
        Initialize the Document Q&A Agent.
        
        Args:
            llm_provider: LLM provider to use ('openai' or 'gemini')
            documents_path: Path to documents directory
        """
        self.llm_client = get_llm_client(llm_provider)
        self.pdf_processor = BatchPDFProcessor()
        self.indexer = DocumentIndexer(self.llm_client)
        self.documents_path = documents_path or config.documents_path
        self.conversation_history = []
        
        # Create necessary directories
        os.makedirs(self.documents_path, exist_ok=True)
        os.makedirs(config.cache_path, exist_ok=True)
        os.makedirs(config.vector_db_path, exist_ok=True)
        
        logger.info("Document Q&A Agent initialized")
    
    def ingest_documents(self, force_reindex: bool = False) -> Dict[str, str]:
        """
        Ingest and index all PDF documents in the documents directory.
        
        Args:
            force_reindex: Whether to force re-indexing of all documents
            
        Returns:
            Dictionary mapping file names to document IDs
        """
        logger.info(f"Starting document ingestion from: {self.documents_path}")
        
        try:
            # Process all PDFs in the documents directory
            extracted_contents = self.pdf_processor.process_directory(self.documents_path)
            
            if not extracted_contents:
                logger.warning("No PDF documents found to process")
                return {}
            
            indexed_docs = {}
            
            # Index each processed document
            for file_name, content in extracted_contents.items():
                file_path = os.path.join(self.documents_path, file_name)
                doc_id = self.indexer.index_document(file_path, content)
                indexed_docs[file_name] = doc_id
                
                # Save extracted content for caching
                cache_file = os.path.join(config.cache_path, f"{doc_id}.json")
                self.pdf_processor.save_extracted_content(content, cache_file)
            
            # Save the document index
            index_file = os.path.join(config.vector_db_path, "document_index.json")
            self.indexer.save_index(index_file)
            
            logger.info(f"Successfully ingested {len(indexed_docs)} documents")
            return indexed_docs
            
        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            raise
    
    def query(
        self, 
        question: str, 
        query_type: str = "auto",
        context_limit: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query and return an intelligent response.
        
        Args:
            question: User's question
            query_type: Type of query ('lookup', 'summary', 'evaluation', 'auto')
            context_limit: Maximum number of context chunks to use
            include_sources: Whether to include source information
            
        Returns:
            Response dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        try:
            # Determine query type if auto
            if query_type == "auto":
                query_type = self._classify_query(question)
            
            # Search for relevant documents
            relevant_chunks = self.indexer.search_documents(question, top_k=context_limit)
            
            if not relevant_chunks:
                return {
                    'answer': "I couldn't find any relevant information in the indexed documents. Please make sure documents are properly ingested.",
                    'sources': [],
                    'query_type': query_type,
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Generate response based on query type
            response = self._generate_response(question, relevant_chunks, query_type)
            
            # Add conversation to history
            self.conversation_history.append({
                'question': question,
                'answer': response['answer'],
                'query_type': query_type,
                'timestamp': datetime.now().isoformat(),
                'sources_count': len(response.get('sources', []))
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f"An error occurred while processing your query: {str(e)}",
                'sources': [],
                'query_type': query_type,
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _classify_query(self, question: str) -> str:
        """
        Classify the type of query based on keywords and patterns.
        
        Args:
            question: User's question
            
        Returns:
            Query type classification
        """
        question_lower = question.lower()
        
        # Evaluation/metrics queries
        evaluation_keywords = [
            'accuracy', 'f1-score', 'precision', 'recall', 'performance', 
            'metrics', 'results', 'evaluation', 'score', 'benchmark'
        ]
        if any(keyword in question_lower for keyword in evaluation_keywords):
            return "evaluation"
        
        # Summary queries
        summary_keywords = [
            'summarize', 'summary', 'overview', 'main points', 'key insights',
            'methodology', 'approach', 'findings', 'conclusions'
        ]
        if any(keyword in question_lower for keyword in summary_keywords):
            return "summary"
        
        # Direct lookup queries
        lookup_keywords = [
            'what is', 'what are', 'define', 'definition', 'explain',
            'describe', 'who', 'when', 'where', 'how'
        ]
        if any(keyword in question_lower for keyword in lookup_keywords):
            return "lookup"
        
        # Default to lookup
        return "lookup"
    
    def _generate_response(
        self, 
        question: str, 
        relevant_chunks: List[Dict[str, Any]], 
        query_type: str
    ) -> Dict[str, Any]:
        """
        Generate a response based on the query type and relevant context.
        
        Args:
            question: User's question
            relevant_chunks: Relevant document chunks
            query_type: Type of query
            
        Returns:
            Response dictionary
        """
        # Build context from relevant chunks
        context_text = self._build_context(relevant_chunks)
        
        # Generate appropriate system message based on query type
        system_message = self._get_system_message(query_type)
        
        # Create the prompt
        prompt = f"""
Context from documents:
{context_text}

User Question: {question}

Please provide a comprehensive answer based on the context above. If you cannot find the information in the context, please state that clearly.
"""
        
        # Add specific instructions based on query type
        if query_type == "evaluation":
            prompt += "\nFocus on extracting specific metrics, numbers, and evaluation results."
        elif query_type == "summary":
            prompt += "\nProvide a structured summary highlighting the key points and insights."
        elif query_type == "lookup":
            prompt += "\nProvide a direct and specific answer to the question."
        
        try:
            # Generate response using LLM
            answer = self.llm_client.generate_response(
                prompt=prompt,
                system_message=system_message,
                max_tokens=config.max_tokens,
                temperature=config.temperature
            )
            
            # Calculate confidence based on relevance scores
            confidence = self._calculate_confidence(relevant_chunks)
            
            # Prepare sources
            sources = []
            if relevant_chunks:
                for chunk in relevant_chunks[:3]:  # Top 3 sources
                    sources.append({
                        'document': chunk.get('title', 'Unknown Document'),
                        'file_path': chunk.get('file_path', ''),
                        'type': chunk.get('type', 'text'),
                        'similarity': chunk.get('similarity', 0.0),
                        'excerpt': chunk.get('text', '')[:200] + '...' if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
                    })
            
            return {
                'answer': answer,
                'sources': sources,
                'query_type': query_type,
                'confidence': confidence,
                'context_used': len(relevant_chunks),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _build_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Build context string from relevant chunks."""
        context_parts = []
        
        for i, chunk in enumerate(relevant_chunks):
            doc_title = chunk.get('title', 'Unknown Document')
            chunk_type = chunk.get('type', 'text')
            text = chunk.get('text', '')
            
            context_part = f"""
Document {i+1}: {doc_title}
Type: {chunk_type}
Content: {text}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _get_system_message(self, query_type: str) -> str:
        """Get system message based on query type."""
        base_message = """You are an expert AI assistant specialized in analyzing academic and research documents. 
You have access to extracted content from PDF documents and should provide accurate, well-structured responses based on the given context."""
        
        if query_type == "evaluation":
            return base_message + """
Focus on extracting and presenting specific numerical results, performance metrics, evaluation scores, and quantitative findings. 
Be precise about the numbers and clearly state the conditions under which they were obtained."""
        
        elif query_type == "summary":
            return base_message + """
Provide comprehensive summaries that capture the main ideas, methodologies, key findings, and conclusions. 
Structure your response with clear sections and highlight the most important insights."""
        
        elif query_type == "lookup":
            return base_message + """
Provide direct, specific answers to factual questions. Be concise but complete, and clearly indicate if information is not available in the provided context."""
        
        return base_message
    
    def _calculate_confidence(self, relevant_chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on relevance of chunks."""
        if not relevant_chunks:
            return 0.0
        
        # Average similarity score of top chunks
        similarities = [chunk.get('similarity', 0.0) for chunk in relevant_chunks[:3]]
        avg_similarity = sum(similarities) / len(similarities)
        
        # Normalize to 0-1 range (assuming similarity scores are between -1 and 1)
        confidence = max(0.0, min(1.0, (avg_similarity + 1) / 2))
        
        return round(confidence, 2)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed documents."""
        docs = self.indexer.list_documents()
        
        stats = {
            'total_documents': len(docs),
            'documents': [],
            'total_chunks': 0,
            'total_sections': 0
        }
        
        for doc_id in docs:
            doc = self.indexer.get_document(doc_id)
            if doc:
                doc_stats = {
                    'doc_id': doc_id,
                    'title': doc['title'] or 'Unknown Title',
                    'file_path': doc['file_path'],
                    'text_chunks': len(doc['text_chunks']),
                    'sections': len(doc['section_chunks']),
                    'tables': len(doc.get('tables', [])),
                    'figures': len(doc.get('figures', [])),
                    'references': len(doc.get('references', [])),
                    'equations': len(doc.get('equations', [])),
                    'authors': doc.get('authors', [])
                }
                stats['documents'].append(doc_stats)
                stats['total_chunks'] += doc_stats['text_chunks']
                stats['total_sections'] += doc_stats['sections']
        
        return stats
    
    def extract_specific_content(self, doc_id: str, content_type: str) -> Any:
        """
        Extract specific content from a document.
        
        Args:
            doc_id: Document identifier
            content_type: Type of content ('tables', 'figures', 'equations', 'references')
            
        Returns:
            Requested content
        """
        doc = self.indexer.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document not found: {doc_id}")
        
        content_map = {
            'tables': doc.get('tables', []),
            'figures': doc.get('figures', []),
            'equations': doc.get('equations', []),
            'references': doc.get('references', []),
            'abstract': doc.get('abstract'),
            'title': doc.get('title'),
            'authors': doc.get('authors', []),
            'metadata': doc.get('metadata', {})
        }
        
        if content_type not in content_map:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        return content_map[content_type]
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the agent."""
        try:
            # Test LLM client
            llm_status = "healthy"
            try:
                test_response = self.llm_client.generate_response("Hello", max_tokens=10)
                if not test_response:
                    llm_status = "unhealthy"
            except Exception as e:
                llm_status = f"error: {str(e)}"
            
            # Check document status
            docs_count = len(self.indexer.list_documents())
            
            # Check directories
            directories_status = {
                'documents_path': os.path.exists(self.documents_path),
                'cache_path': os.path.exists(config.cache_path),
                'vector_db_path': os.path.exists(config.vector_db_path)
            }
            
            return {
                'status': 'healthy' if llm_status == 'healthy' and docs_count > 0 else 'partially_healthy',
                'llm_client': llm_status,
                'indexed_documents': docs_count,
                'directories': directories_status,
                'conversation_history_length': len(self.conversation_history),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

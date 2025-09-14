"""
Test suite for the AI Document Agent.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from processors.pdf_processor import PDFProcessor, ExtractedContent
from processors.content_extractor import ContentChunker, DocumentIndexer
from agents.document_agent import DocumentQAAgent


class TestConfig:
    """Test configuration management."""
    
    def test_config_initialization(self):
        """Test config initialization."""
        config = Config()
        assert hasattr(config, 'default_llm_provider')
        assert hasattr(config, 'chunk_size')
        assert hasattr(config, 'temperature')
    
    def test_config_validation(self):
        """Test config validation."""
        # This will depend on environment variables
        config = Config()
        # Basic structure tests
        assert config.chunk_size > 0
        assert 0.0 <= config.temperature <= 1.0


class TestContentChunker:
    """Test content chunking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = ContentChunker(chunk_size=100, chunk_overlap=20)
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "This is a test document. " * 10
        chunks = self.chunker.chunk_text(text)
        
        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)
        assert all('chunk_id' in chunk for chunk in chunks)
    
    def test_chunk_text_preserve_structure(self):
        """Test structure-preserving chunking."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = self.chunker.chunk_text(text, preserve_structure=True)
        
        assert len(chunks) > 0
        # Should preserve paragraph boundaries
        for chunk in chunks:
            assert '\n\n' not in chunk['text'] or len(chunk['text'].split('\n\n')) <= 3
    
    def test_chunk_overlap(self):
        """Test chunk overlap functionality."""
        text = "A" * 200
        chunker = ContentChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_text(text, preserve_structure=False)
        
        if len(chunks) > 1:
            # Check that there's overlap between consecutive chunks
            first_chunk_end = chunks[0]['text'][-10:]
            second_chunk_start = chunks[1]['text'][:10]
            # Some overlap should exist


class TestPDFProcessor:
    """Test PDF processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = PDFProcessor()
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        assert self.processor.supported_formats == ['.pdf']
    
    def test_academic_structure_parsing(self):
        """Test academic paper structure parsing."""
        text = """
        Title: Neural Networks for AI
        
        Abstract: This paper presents a novel approach to neural networks.
        
        1. Introduction
        This section introduces the topic.
        
        2. Methodology
        Our methodology involves...
        
        References
        [1] Smith, J. (2020). AI Research.
        """
        
        structure = self.processor._parse_academic_structure(text)
        
        assert structure['title'] is not None
        assert structure['abstract'] is not None
        assert len(structure['sections']) > 0
        assert len(structure['references']) > 0
    
    def test_equation_extraction(self):
        """Test equation extraction."""
        text = "The formula is $E = mc^2$ and also $$\\sum_{i=1}^n x_i$$"
        equations = self.processor._extract_equations(text)
        
        assert len(equations) >= 1


class TestDocumentAgent:
    """Test the main document agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the LLM client to avoid API calls in tests
        with patch('utils.llm_client.get_llm_client') as mock_client:
            mock_llm = Mock()
            mock_llm.generate_response.return_value = "This is a test response."
            mock_llm.generate_embeddings.return_value = [0.1] * 768
            mock_client.return_value = mock_llm
            
            self.agent = DocumentQAAgent()
            self.agent.llm_client = mock_llm
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.agent is not None
        assert hasattr(self.agent, 'llm_client')
        assert hasattr(self.agent, 'indexer')
        assert hasattr(self.agent, 'pdf_processor')
    
    def test_query_classification(self):
        """Test query type classification."""
        # Test different query types
        evaluation_query = "What is the F1-score in this paper?"
        summary_query = "Summarize the main findings"
        lookup_query = "What is the definition of neural networks?"
        
        eval_type = self.agent._classify_query(evaluation_query)
        summary_type = self.agent._classify_query(summary_query)
        lookup_type = self.agent._classify_query(lookup_query)
        
        assert eval_type == "evaluation"
        assert summary_type == "summary"
        assert lookup_type == "lookup"
    
    def test_health_check(self):
        """Test health check functionality."""
        health = self.agent.health_check()
        
        assert 'status' in health
        assert 'llm_client' in health
        assert 'indexed_documents' in health
        assert 'timestamp' in health


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_mock(self):
        """Test end-to-end functionality with mocked components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock extracted content
            mock_content = ExtractedContent(
                text="This is a test document with sample content.",
                title="Test Document",
                authors=["Test Author"],
                abstract="This is a test abstract.",
                sections=[
                    {"title": "Introduction", "content": "Introduction content"},
                    {"title": "Methodology", "content": "Methodology content"}
                ],
                tables=[],
                figures=[],
                references=["Reference 1", "Reference 2"],
                equations=["E = mc^2"],
                metadata={"pages": 10}
            )
            
            # Mock the LLM client
            with patch('utils.llm_client.get_llm_client') as mock_client:
                mock_llm = Mock()
                mock_llm.generate_response.return_value = "This is a comprehensive answer based on the document content."
                mock_llm.generate_embeddings.return_value = [0.1] * 768
                mock_client.return_value = mock_llm
                
                # Initialize agent
                agent = DocumentQAAgent(documents_path=temp_dir)
                
                # Index the mock content
                doc_id = agent.indexer.index_document("test.pdf", mock_content)
                
                # Test querying
                response = agent.query("What is this document about?")
                
                assert response['answer'] is not None
                assert response['query_type'] in ['lookup', 'summary', 'evaluation']
                assert 'confidence' in response
                assert 'timestamp' in response


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v'])


if __name__ == "__main__":
    run_tests()

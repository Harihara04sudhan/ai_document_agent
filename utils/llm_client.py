"""
LLM client implementation for Google Gemini API.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from .config import config


logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the given text."""
        pass


class GeminiClient(BaseLLMClient):
    """Google Gemini API client."""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        if not genai:
            raise ImportError("Google GenerativeAI library not installed. Install with: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.embedding_model = "models/embedding-001"
    
    def generate_response(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Generate a response using Gemini's generate content API."""
        try:
            full_prompt = prompt
            if system_message:
                full_prompt = f"System: {system_message}\n\nUser: {prompt}"
            
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens or config.max_tokens,
                temperature=temperature or config.temperature,
            )
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            raise
    
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using Gemini's embedding API."""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Error generating Gemini embeddings: {e}")
            # Fallback: return zero embeddings if embedding fails
            return [0.0] * 768


class LLMClientFactory:
    """Factory class for creating LLM clients."""
    
    @staticmethod
    def create_client(provider: str = None) -> BaseLLMClient:
        """Create an LLM client based on the provider."""
        provider = provider or config.default_llm_provider
        
        if provider == "gemini":
            if not config.gemini_api_key:
                raise ValueError("Gemini API key not found")
            return GeminiClient(config.gemini_api_key, config.gemini_model)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Only 'gemini' is supported.")


class MultiModalLLMClient:
    """Wrapper for multi-modal capabilities."""
    
    def __init__(self, client: BaseLLMClient):
        self.client = client
    
    def analyze_image_with_text(
        self, 
        image_path: str, 
        text_prompt: str, 
        system_message: Optional[str] = None
    ) -> str:
        """Analyze an image along with text (for advanced PDF processing)."""
        # This would be implemented when we add vision capabilities
        # For now, we'll focus on text-based processing
        logger.warning("Multi-modal image analysis not yet implemented")
        return self.client.generate_response(text_prompt, system_message)
    
    def extract_table_structure(self, table_text: str) -> Dict[str, Any]:
        """Extract and structure table data."""
        system_message = """You are an expert at analyzing table structures. 
        Given table text, extract the structure and return it in a structured format."""
        
        prompt = f"""
        Analyze the following table text and extract its structure:
        
        {table_text}
        
        Please return the table structure in a JSON format with:
        - headers: list of column headers
        - rows: list of row data
        - metadata: any additional information about the table
        """
        
        return self.client.generate_response(prompt, system_message)


# Global client instance
def get_llm_client(provider: str = None) -> BaseLLMClient:
    """Get a configured LLM client."""
    return LLMClientFactory.create_client(provider)

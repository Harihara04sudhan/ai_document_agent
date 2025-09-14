"""
Configuration management for the AI Document Agent.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
import logging


class Config:
    """Configuration class for managing environment variables and settings."""
    
    def __init__(self):
        load_dotenv()
        self._validate_config()
    
    # API Configuration
    @property
    def gemini_api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", "")
    
    @property
    def default_llm_provider(self) -> str:
        return os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    
    @property
    def gemini_model(self) -> str:
        return os.getenv("GEMINI_MODEL", "gemini-pro")
    
    # LLM Parameters
    @property
    def max_tokens(self) -> int:
        return int(os.getenv("MAX_TOKENS", "4096"))
    
    @property
    def temperature(self) -> float:
        return float(os.getenv("TEMPERATURE", "0.3"))
    
    @property
    def chunk_size(self) -> int:
        return int(os.getenv("CHUNK_SIZE", "1000"))
    
    @property
    def chunk_overlap(self) -> int:
        return int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Storage Configuration
    @property
    def vector_db_path(self) -> str:
        return os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    
    @property
    def documents_path(self) -> str:
        return os.getenv("DOCUMENTS_PATH", "./documents")
    
    @property
    def cache_path(self) -> str:
        return os.getenv("CACHE_PATH", "./data/cache")
    
    # Logging Configuration
    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def log_file(self) -> str:
        return os.getenv("LOG_FILE", "./logs/app.log")
    
    # Arxiv Configuration
    @property
    def arxiv_max_results(self) -> int:
        return int(os.getenv("ARXIV_MAX_RESULTS", "10"))
    
    def _validate_config(self) -> None:
        """Validate critical configuration settings."""
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set")
        
        if self.default_llm_provider != "gemini":
            raise ValueError("Only Gemini provider is supported in this configuration")
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary."""
        return {
            "default_llm_provider": self.default_llm_provider,
            "gemini_model": self.gemini_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vector_db_path": self.vector_db_path,
            "documents_path": self.documents_path,
            "cache_path": self.cache_path,
            "log_level": self.log_level,
            "arxiv_max_results": self.arxiv_max_results
        }


def setup_logging(config: Config) -> None:
    """Set up logging configuration."""
    os.makedirs(os.path.dirname(config.log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )


# Global configuration instance
config = Config()

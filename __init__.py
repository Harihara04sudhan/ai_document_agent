"""
AI Document Agent - Enterprise-ready document Q&A system.
"""

__version__ = "1.0.0"
__author__ = "AI Engineer"
__description__ = "Enterprise-ready AI agent for document processing and intelligent Q&A"

from .agents.document_agent import DocumentQAAgent
from .agents.arxiv_agent import ArxivAgent
from .utils.config import config

__all__ = [
    "DocumentQAAgent",
    "ArxivAgent", 
    "config"
]

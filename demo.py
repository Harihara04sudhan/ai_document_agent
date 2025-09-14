#!/usr/bin/env python3
"""
Demo script showing the capabilities of the AI Document Agent.
"""

import os
import sys
import time
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.document_agent import DocumentQAAgent
from agents.arxiv_agent import ArxivAgent
from utils.config import config


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n📌 {title}")
    print("-" * 40)


def demo_document_agent():
    """Demonstrate document agent capabilities."""
    print_header("AI Document Agent Demo")
    
    try:
        # Initialize the agent
        print("🤖 Initializing AI Document Agent...")
        agent = DocumentQAAgent()
        
        print_section("Health Check")
        health = agent.health_check()
        print(f"Status: {health['status']}")
        print(f"Indexed Documents: {health['indexed_documents']}")
        
        if health['indexed_documents'] == 0:
            print("\n⚠️  No documents are indexed yet.")
            print("To use the document Q&A features:")
            print("1. Add PDF files to the 'documents' folder")
            print("2. Run: python ingest_documents.py")
            print("3. Then run this demo again")
            return
        
        print_section("Document Statistics")
        stats = agent.get_document_statistics()
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        
        if stats['documents']:
            print("\nDocument Details:")
            for doc in stats['documents'][:3]:  # Show first 3 documents
                print(f"  • {doc['title'] or 'Unknown Title'}")
                print(f"    Chunks: {doc['text_chunks']}, Tables: {doc['tables']}, Figures: {doc['figures']}")
        
        print_section("Sample Queries")
        
        # Demo different query types
        sample_queries = [
            ("What are the main conclusions?", "lookup"),
            ("Summarize the key findings", "summary"),
            ("What evaluation metrics are reported?", "evaluation")
        ]
        
        for query, query_type in sample_queries:
            print(f"\n💭 Query: {query}")
            print(f"Expected Type: {query_type}")
            
            try:
                response = agent.query(query, context_limit=3)
                
                print(f"🤖 Response ({response['query_type']}):")
                print(f"{response['answer'][:200]}...")
                print(f"Confidence: {response['confidence']}")
                print(f"Sources: {len(response.get('sources', []))}")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
            
            time.sleep(1)  # Brief pause between queries
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_arxiv_agent():
    """Demonstrate Arxiv agent capabilities."""
    print_header("Arxiv Agent Demo")
    
    try:
        # Initialize agents
        print("🤖 Initializing Arxiv Agent...")
        document_agent = DocumentQAAgent()
        arxiv_agent = ArxivAgent(document_agent.llm_client)
        
        print_section("Health Check")
        health = arxiv_agent.health_check()
        print(f"Status: {health['status']}")
        print(f"Arxiv API: {health['arxiv_api']}")
        
        if health['status'] != 'healthy':
            print("⚠️  Arxiv agent not fully functional. Some features may not work.")
            return
        
        print_section("Paper Search Demo")
        
        # Demo paper search
        search_queries = [
            "transformer architecture",
            "neural machine translation",
            "computer vision"
        ]
        
        for query in search_queries[:1]:  # Just do one search for demo
            print(f"\n🔍 Searching for: {query}")
            
            try:
                papers = arxiv_agent.search_papers(query, max_results=3)
                
                if papers:
                    print(f"Found {len(papers)} papers:")
                    for i, paper in enumerate(papers, 1):
                        print(f"\n  {i}. {paper['title']}")
                        print(f"     Authors: {', '.join(paper['authors'][:2])}")
                        print(f"     Published: {paper['published'][:10] if paper['published'] else 'Unknown'}")
                        print(f"     Categories: {', '.join(paper['categories'][:2])}")
                        print(f"     Abstract: {paper['abstract'][:150]}...")
                else:
                    print("No papers found.")
                    
            except Exception as e:
                print(f"❌ Search failed: {e}")
    
    except Exception as e:
        print(f"❌ Arxiv demo failed: {e}")


def demo_integration():
    """Demonstrate integration capabilities."""
    print_header("Integration Demo")
    
    print_section("Configuration")
    settings = config.get_all_settings()
    for key, value in list(settings.items())[:5]:  # Show first 5 settings
        if 'key' not in key.lower():  # Don't show API keys
            print(f"  {key}: {value}")
    
    print_section("Enterprise Features")
    print("✅ Multi-modal content extraction")
    print("✅ Context-aware responses")
    print("✅ Secure API key management")
    print("✅ Comprehensive error handling")
    print("✅ Detailed logging and monitoring")
    print("✅ Caching and optimization")
    print("✅ Arxiv integration (bonus)")
    
    print_section("Supported Query Types")
    print("1. Direct Content Lookup")
    print("   Example: 'What is the conclusion of Paper X?'")
    print("\n2. Summarization")
    print("   Example: 'Summarize the methodology section'")
    print("\n3. Evaluation Results Extraction")
    print("   Example: 'What are the accuracy scores reported?'")


def main():
    """Main demo function."""
    print("🚀 AI Document Agent - Enterprise Demo")
    print("=====================================")
    
    # Check environment
    if not os.path.exists('.env'):
        print("\n⚠️  Warning: No .env file found!")
        print("Create a .env file with your API keys for full functionality.")
        print("See .env.example for the template.")
    
    # Run demos
    demo_integration()
    demo_document_agent()
    demo_arxiv_agent()
    
    print_header("Demo Complete")
    print("🎯 To start using the agent:")
    print("1. Add PDF documents to the 'documents' folder")
    print("2. Run: python ingest_documents.py")
    print("3. Run: python main.py --interactive")
    print("\n📚 For Arxiv features:")
    print("   python main.py --arxiv 'your search query'")
    print("\n🔍 For single queries:")
    print("   python main.py --query 'your question'")


if __name__ == "__main__":
    main()

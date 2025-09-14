#!/usr/bin/env python3
"""
Main entry point for the AI Document Q&A Agent.

Enterprise-ready AI agent for document processing and intelligent Q&A.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.document_agent import DocumentQAAgent
from agents.arxiv_agent import ArxivAgent
from utils.config import config, setup_logging


def setup_environment():
    """Set up the application environment."""
    # Set up logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        logger.warning("No .env file found. Please create one using .env.example as template")
        print("⚠️  Warning: No .env file found!")
        print("Please create a .env file with your API keys using .env.example as template")
        return False
    
    # Validate configuration
    try:
        config._validate_config()
        logger.info("Configuration validated successfully")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        print(f"❌ Configuration Error: {e}")
        return False


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="AI Document Q&A Agent - Enterprise-ready document processing and Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ingest                    # Ingest documents from documents/ folder
  python main.py --query "What is the main conclusion?"
  python main.py --interactive               # Start interactive mode
  python main.py --arxiv "neural networks"   # Search Arxiv for papers
  python main.py --health                    # Check system health
  python main.py --stats                     # Show document statistics
        """
    )
    
    # Action arguments
    parser.add_argument('--ingest', action='store_true', 
                       help='Ingest and index PDF documents')
    parser.add_argument('--query', type=str, 
                       help='Process a single query')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive Q&A mode')
    parser.add_argument('--arxiv', type=str,
                       help='Search Arxiv for papers')
    parser.add_argument('--health', action='store_true',
                       help='Perform system health check')
    parser.add_argument('--stats', action='store_true',
                       help='Show document statistics')
    
    # Configuration arguments
    parser.add_argument('--llm-provider', choices=['gemini'],
                       help='LLM provider to use (gemini only)')
    parser.add_argument('--documents-path', type=str,
                       help='Path to documents directory')
    parser.add_argument('--max-results', type=int, default=5,
                       help='Maximum number of results to return')
    parser.add_argument('--force-reindex', action='store_true',
                       help='Force re-indexing of all documents')
    
    args = parser.parse_args()
    
    # Set up environment
    if not setup_environment():
        sys.exit(1)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting AI Document Q&A Agent")
    
    try:
        # Initialize the agents
        print("🤖 Initializing AI Document Q&A Agent...")
        
        document_agent = DocumentQAAgent(
            llm_provider=args.llm_provider,
            documents_path=args.documents_path
        )
        
        arxiv_agent = ArxivAgent(document_agent.llm_client)
        
        print("✅ Agent initialized successfully!")
        
        # Handle different actions
        if args.health:
            handle_health_check(document_agent, arxiv_agent)
            
        elif args.ingest:
            handle_document_ingestion(document_agent, args.force_reindex)
            
        elif args.stats:
            handle_statistics(document_agent)
            
        elif args.query:
            handle_single_query(document_agent, args.query, args.max_results)
            
        elif args.arxiv:
            handle_arxiv_search(arxiv_agent, args.arxiv, args.max_results)
            
        elif args.interactive:
            handle_interactive_mode(document_agent, arxiv_agent)
            
        else:
            # Default: show help and basic info
            parser.print_help()
            print("\n" + "="*60)
            print("📊 Quick Status Check:")
            health_info = document_agent.health_check()
            print(f"System Status: {health_info['status']}")
            print(f"Indexed Documents: {health_info['indexed_documents']}")
            print("="*60)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


def handle_health_check(document_agent: DocumentQAAgent, arxiv_agent: ArxivAgent):
    """Handle health check command."""
    print("\n🔍 Performing Health Check...")
    
    # Document agent health check
    doc_health = document_agent.health_check()
    print(f"\n📄 Document Agent:")
    print(f"  Status: {doc_health['status']}")
    print(f"  LLM Client: {doc_health['llm_client']}")
    print(f"  Indexed Documents: {doc_health['indexed_documents']}")
    print(f"  Directories: {doc_health['directories']}")
    
    # Arxiv agent health check
    arxiv_health = arxiv_agent.health_check()
    print(f"\n📚 Arxiv Agent:")
    print(f"  Status: {arxiv_health['status']}")
    print(f"  Arxiv API: {arxiv_health['arxiv_api']}")
    print(f"  LLM Available: {arxiv_health['llm_client_available']}")
    
    # Configuration info
    print(f"\n⚙️  Configuration:")
    settings = config.get_all_settings()
    for key, value in settings.items():
        if 'key' not in key.lower():  # Don't show API keys
            print(f"  {key}: {value}")


def handle_document_ingestion(document_agent: DocumentQAAgent, force_reindex: bool):
    """Handle document ingestion command."""
    print("\n📥 Starting Document Ingestion...")
    
    try:
        indexed_docs = document_agent.ingest_documents(force_reindex=force_reindex)
        
        if indexed_docs:
            print(f"✅ Successfully ingested {len(indexed_docs)} documents:")
            for file_name, doc_id in indexed_docs.items():
                print(f"  • {file_name} → {doc_id}")
        else:
            print("⚠️  No documents found to ingest.")
            print(f"Please add PDF files to: {document_agent.documents_path}")
            
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")


def handle_statistics(document_agent: DocumentQAAgent):
    """Handle statistics command."""
    print("\n📊 Document Statistics:")
    
    try:
        stats = document_agent.get_document_statistics()
        
        print(f"\nTotal Documents: {stats['total_documents']}")
        print(f"Total Text Chunks: {stats['total_chunks']}")
        print(f"Total Sections: {stats['total_sections']}")
        
        if stats['documents']:
            print(f"\n📑 Document Details:")
            for doc in stats['documents']:
                print(f"\n  Document: {doc['title']}")
                print(f"    File: {os.path.basename(doc['file_path'])}")
                print(f"    Chunks: {doc['text_chunks']}")
                print(f"    Sections: {doc['sections']}")
                print(f"    Tables: {doc['tables']}")
                print(f"    Figures: {doc['figures']}")
                if doc['authors']:
                    print(f"    Authors: {', '.join(doc['authors'][:3])}")
        else:
            print("No documents indexed yet. Run with --ingest to index documents.")
            
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")


def handle_single_query(document_agent: DocumentQAAgent, query: str, max_results: int):
    """Handle single query command."""
    print(f"\n💭 Processing Query: {query}")
    
    try:
        response = document_agent.query(
            question=query,
            context_limit=max_results,
            include_sources=True
        )
        
        print(f"\n🤖 Response ({response['query_type']} query):")
        print(f"{response['answer']}")
        
        if response['sources']:
            print(f"\n📚 Sources (confidence: {response['confidence']}):")
            for i, source in enumerate(response['sources'], 1):
                print(f"  {i}. {source['document']}")
                print(f"     Type: {source['type']}, Similarity: {source['similarity']:.3f}")
                print(f"     Excerpt: {source['excerpt']}")
                print()
        else:
            print("\n⚠️  No relevant sources found.")
            
    except Exception as e:
        print(f"❌ Query failed: {e}")


def handle_arxiv_search(arxiv_agent: ArxivAgent, query: str, max_results: int):
    """Handle Arxiv search command."""
    print(f"\n🔍 Searching Arxiv for: {query}")
    
    try:
        papers = arxiv_agent.search_papers(query, max_results)
        
        if papers:
            print(f"\n📄 Found {len(papers)} papers:")
            for i, paper in enumerate(papers, 1):
                print(f"\n  {i}. {paper['title']}")
                print(f"     Authors: {', '.join(paper['authors'][:3])}")
                print(f"     Published: {paper['published'][:10] if paper['published'] else 'Unknown'}")
                print(f"     Categories: {', '.join(paper['categories'])}")
                print(f"     URL: {paper['entry_url']}")
                print(f"     Abstract: {paper['abstract'][:200]}...")
        else:
            print("No papers found for your query.")
            
    except Exception as e:
        print(f"❌ Arxiv search failed: {e}")


def handle_interactive_mode(document_agent: DocumentQAAgent, arxiv_agent: ArxivAgent):
    """Handle interactive Q&A mode."""
    print("\n🎯 Interactive Q&A Mode")
    print("Type 'help' for commands, 'quit' to exit")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n💭 Your question: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
                
            elif user_input.lower() == 'help':
                show_interactive_help()
                continue
                
            elif user_input.lower() == 'stats':
                handle_statistics(document_agent)
                continue
                
            elif user_input.lower() == 'health':
                handle_health_check(document_agent, arxiv_agent)
                continue
                
            elif user_input.lower().startswith('arxiv:'):
                query = user_input[6:].strip()
                if query:
                    handle_arxiv_search(arxiv_agent, query, 3)
                continue
                
            # Regular document query
            response = document_agent.query(user_input, include_sources=True)
            
            print(f"\n🤖 Answer ({response['query_type']} • confidence: {response['confidence']}):")
            print(response['answer'])
            
            if response['sources']:
                print(f"\n📚 Top sources:")
                for i, source in enumerate(response['sources'][:2], 1):
                    print(f"  {i}. {source['document']} (similarity: {source['similarity']:.3f})")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def show_interactive_help():
    """Show help for interactive mode."""
    print("""
🎯 Interactive Mode Commands:
  
  Document Queries:
    Ask any question about your documents
    
  Special Commands:
    help          - Show this help message
    stats         - Show document statistics
    health        - Perform health check
    arxiv: <query> - Search Arxiv papers
    quit/exit     - Exit the program
    
  Query Types:
    Direct lookup: "What is the conclusion of Paper X?"
    Summarization: "Summarize the methodology"
    Evaluation: "What are the accuracy scores?"
    """)


if __name__ == "__main__":
    main()

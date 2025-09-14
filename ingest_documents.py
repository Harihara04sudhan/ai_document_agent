#!/usr/bin/env python3
"""
Document ingestion script for batch processing PDFs.
"""

import os
import sys
import argparse
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.document_agent import DocumentQAAgent
from utils.config import config, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF documents for the AI Document Agent")
    
    parser.add_argument('--documents-path', type=str, default=None,
                       help='Path to documents directory')
    parser.add_argument('--force-reindex', action='store_true',
                       help='Force re-indexing of all documents')
    parser.add_argument('--llm-provider', choices=['gemini'], default=None,
                       help='LLM provider to use (gemini only)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        os.environ['LOG_LEVEL'] = 'DEBUG'
    
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    try:
        print("🤖 Initializing AI Document Agent for ingestion...")
        
        # Initialize the agent
        agent = DocumentQAAgent(
            llm_provider=args.llm_provider,
            documents_path=args.documents_path
        )
        
        print(f"📁 Documents path: {agent.documents_path}")
        
        # Check if documents exist
        if not os.path.exists(agent.documents_path):
            print(f"❌ Documents directory not found: {agent.documents_path}")
            sys.exit(1)
        
        # Look for both PDF and text files
        document_files = [f for f in os.listdir(agent.documents_path) 
                         if f.lower().endswith(('.pdf', '.txt'))]
        
        if not document_files:
            print(f"⚠️  No supported document files found in: {agent.documents_path}")
            print("Please add PDF or text files to the documents directory.")
            sys.exit(1)
        
        print(f"📄 Found {len(document_files)} document files to process:")
        for doc_file in document_files:
            file_type = "PDF" if doc_file.lower().endswith('.pdf') else "Text"
            print(f"  • {doc_file} ({file_type})")
        
        # Start ingestion
        print("\n🔄 Starting document ingestion...")
        indexed_docs = agent.ingest_documents(force_reindex=args.force_reindex)
        
        if indexed_docs:
            print(f"\n✅ Successfully ingested {len(indexed_docs)} documents:")
            
            # Show statistics
            stats = agent.get_document_statistics()
            
            print(f"\n📊 Statistics:")
            print(f"  Total documents: {stats['total_documents']}")
            print(f"  Total text chunks: {stats['total_chunks']}")
            print(f"  Total sections: {stats['total_sections']}")
            
            print(f"\n📑 Document details:")
            for doc in stats['documents']:
                print(f"\n  📄 {doc['title'] or 'Unknown Title'}")
                print(f"     File: {os.path.basename(doc['file_path'])}")
                print(f"     Text chunks: {doc['text_chunks']}")
                print(f"     Sections: {doc['sections']}")
                print(f"     Tables: {doc['tables']}")
                print(f"     Figures: {doc['figures']}")
                print(f"     Equations: {doc['equations']}")
                if doc['authors']:
                    print(f"     Authors: {', '.join(doc['authors'][:3])}")
            
            print(f"\n🎯 Ready for queries! Use: python main.py --interactive")
        
        else:
            print("❌ No documents were successfully ingested.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Ingestion interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"❌ Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Streamlit web interface for the AI Document Agent.
"""

import streamlit as st
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from agents.document_agent import DocumentQAAgent
    from agents.arxiv_agent import ArxivAgent
    from utils.config import config
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed and the environment is properly configured.")
    st.stop()


# Page config
st.set_page_config(
    page_title="AI Document Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def initialize_agents():
    """Initialize agents with caching."""
    try:
        document_agent = DocumentQAAgent()
        arxiv_agent = ArxivAgent(document_agent.llm_client)
        return document_agent, arxiv_agent
    except Exception as e:
        st.error(f"Failed to initialize agents: {e}")
        return None, None


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    st.sidebar.title("🤖 AI Document Agent")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["Document Q&A", "Arxiv Search", "Document Management", "System Status"]
    )
    
    # Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings")
    
    # Query settings
    max_results = st.sidebar.slider("Max Results", 1, 10, 5)
    include_sources = st.sidebar.checkbox("Include Sources", True)
    
    # LLM settings
    temperature = st.sidebar.slider("Response Creativity", 0.0, 1.0, 0.3, 0.1)
    
    return page, {
        'max_results': max_results,
        'include_sources': include_sources,
        'temperature': temperature
    }


def render_document_qa_page(agent: DocumentQAAgent, settings: Dict[str, Any]):
    """Render the Document Q&A page."""
    st.title("📄 Document Q&A")
    
    # Check if documents are indexed
    stats = agent.get_document_statistics()
    
    if stats['total_documents'] == 0:
        st.warning("⚠️ No documents are indexed yet!")
        st.info("Please add PDF files to the documents folder and run the ingestion process.")
        
        if st.button("🔄 Run Document Ingestion"):
            with st.spinner("Ingesting documents..."):
                try:
                    indexed_docs = agent.ingest_documents()
                    if indexed_docs:
                        st.success(f"✅ Successfully ingested {len(indexed_docs)} documents!")
                        st.rerun()
                    else:
                        st.error("No documents found to ingest.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
        return
    
    # Display document stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Documents", stats['total_documents'])
    with col2:
        st.metric("📝 Text Chunks", stats['total_chunks'])
    with col3:
        st.metric("📑 Sections", stats['total_sections'])
    
    # Query interface
    st.markdown("---")
    
    # Query input
    query = st.text_area(
        "💭 Ask a question about your documents:",
        placeholder="Example: What are the main conclusions? Summarize the methodology. What evaluation metrics are reported?",
        height=100
    )
    
    # Query type selection
    query_type = st.selectbox(
        "Query Type",
        ["auto", "lookup", "summary", "evaluation"],
        help="Select query type or let the system auto-detect"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🚀 Ask Question", type="primary"):
            if query.strip():
                process_query(agent, query, query_type, settings)
            else:
                st.error("Please enter a question.")
    
    with col2:
        if st.button("🔄 Clear History"):
            agent.clear_conversation_history()
            st.success("Conversation history cleared!")
    
    # Display conversation history
    history = agent.get_conversation_history()
    if history:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        
        for i, entry in enumerate(reversed(history[-5:])):  # Show last 5 entries
            with st.expander(f"Q{len(history)-i}: {entry['question'][:50]}..."):
                st.write("**Question:**", entry['question'])
                st.write("**Answer:**", entry['answer'])
                st.write("**Type:**", entry['query_type'])
                st.write("**Time:**", entry['timestamp'][:19])


def process_query(agent: DocumentQAAgent, query: str, query_type: str, settings: Dict[str, Any]):
    """Process a user query and display results."""
    with st.spinner("🔍 Processing your question..."):
        try:
            response = agent.query(
                question=query,
                query_type=query_type,
                context_limit=settings['max_results'],
                include_sources=settings['include_sources']
            )
            
            # Display response
            st.markdown("---")
            st.subheader("🤖 Answer")
            
            # Show query type and confidence
            col1, col2 = st.columns(2)
            with col1:
                st.badge(f"Query Type: {response['query_type']}")
            with col2:
                confidence_color = "green" if response['confidence'] > 0.7 else "orange" if response['confidence'] > 0.4 else "red"
                st.markdown(f"**Confidence:** :{confidence_color}[{response['confidence']:.2f}]")
            
            # Display answer
            st.write(response['answer'])
            
            # Display sources
            if response['sources'] and settings['include_sources']:
                st.markdown("---")
                st.subheader("📚 Sources")
                
                for i, source in enumerate(response['sources'], 1):
                    with st.expander(f"Source {i}: {source['document']}"):
                        st.write("**Document:**", source['document'])
                        st.write("**Type:**", source['type'])
                        st.write("**Similarity:**", f"{source['similarity']:.3f}")
                        st.write("**Excerpt:**", source['excerpt'])
            
        except Exception as e:
            st.error(f"Query failed: {e}")


def render_arxiv_page(arxiv_agent: ArxivAgent, settings: Dict[str, Any]):
    """Render the Arxiv search page."""
    st.title("📚 Arxiv Paper Search")
    
    # Search interface
    search_query = st.text_input(
        "🔍 Search Arxiv papers:",
        placeholder="Example: transformer architecture, neural networks, machine learning"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        max_papers = st.selectbox("Max Results", [3, 5, 10, 15], index=1)
    with col2:
        sort_by = st.selectbox("Sort By", ["relevance", "lastUpdatedDate", "submittedDate"])
    
    if st.button("🔍 Search Papers", type="primary"):
        if search_query.strip():
            search_arxiv_papers(arxiv_agent, search_query, max_papers, sort_by)
        else:
            st.error("Please enter a search query.")
    
    # Paper recommendations
    st.markdown("---")
    st.subheader("💡 Get Paper Recommendations")
    
    description = st.text_area(
        "Describe the type of papers you're looking for:",
        placeholder="Example: I'm looking for recent papers on transformer architectures for natural language processing",
        height=80
    )
    
    if st.button("🎯 Get Recommendations"):
        if description.strip():
            get_paper_recommendations(arxiv_agent, description)
        else:
            st.error("Please enter a description.")


def search_arxiv_papers(arxiv_agent: ArxivAgent, query: str, max_results: int, sort_by: str):
    """Search Arxiv papers and display results."""
    with st.spinner("🔍 Searching Arxiv..."):
        try:
            papers = arxiv_agent.search_papers(
                query=query,
                max_results=max_results,
                sort_by=sort_by
            )
            
            if papers:
                st.success(f"✅ Found {len(papers)} papers")
                
                for i, paper in enumerate(papers, 1):
                    with st.expander(f"{i}. {paper['title']}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write("**Authors:**", ", ".join(paper['authors'][:3]))
                            if len(paper['authors']) > 3:
                                st.write(f"... and {len(paper['authors']) - 3} more")
                            
                            st.write("**Published:**", paper['published'][:10] if paper['published'] else "Unknown")
                            st.write("**Categories:**", ", ".join(paper['categories']))
                            st.write("**Abstract:**", paper['abstract'][:500] + "..." if len(paper['abstract']) > 500 else paper['abstract'])
                        
                        with col2:
                            st.link_button("📄 PDF", paper['pdf_url'])
                            st.link_button("🔗 Arxiv", paper['entry_url'])
                            
                            if st.button(f"📋 Summarize", key=f"summarize_{i}"):
                                summarize_paper(arxiv_agent, paper['arxiv_id'])
            else:
                st.warning("No papers found for your query.")
                
        except Exception as e:
            st.error(f"Search failed: {e}")


def get_paper_recommendations(arxiv_agent: ArxivAgent, description: str):
    """Get paper recommendations based on description."""
    with st.spinner("🎯 Generating recommendations..."):
        try:
            papers = arxiv_agent.recommend_papers(description, max_results=5)
            
            if papers:
                st.success(f"✅ Found {len(papers)} recommended papers")
                
                for i, paper in enumerate(papers, 1):
                    with st.expander(f"Recommendation {i}: {paper['title']}"):
                        st.write("**Authors:**", ", ".join(paper['authors'][:3]))
                        st.write("**Published:**", paper['published'][:10] if paper['published'] else "Unknown")
                        st.write("**Categories:**", ", ".join(paper['categories']))
                        st.write("**Abstract:**", paper['abstract'][:400] + "...")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.link_button("📄 View PDF", paper['pdf_url'], key=f"pdf_rec_{i}")
                        with col2:
                            st.link_button("🔗 Arxiv Page", paper['entry_url'], key=f"arxiv_rec_{i}")
            else:
                st.warning("No recommendations found.")
                
        except Exception as e:
            st.error(f"Recommendations failed: {e}")


def summarize_paper(arxiv_agent: ArxivAgent, arxiv_id: str):
    """Summarize a paper."""
    with st.spinner("📋 Generating summary..."):
        try:
            summary = arxiv_agent.summarize_paper(arxiv_id)
            
            st.markdown("---")
            st.subheader(f"📋 Summary: {summary['title']}")
            st.write(summary['summary'])
            
        except Exception as e:
            st.error(f"Summarization failed: {e}")


def render_document_management_page(agent: DocumentQAAgent):
    """Render document management page."""
    st.title("📁 Document Management")
    
    # Document statistics
    stats = agent.get_document_statistics()
    
    # Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", stats['total_documents'])
    with col2:
        st.metric("Text Chunks", stats['total_chunks'])
    with col3:
        st.metric("Sections", stats['total_sections'])
    
    # Document list
    st.markdown("---")
    st.subheader("📚 Indexed Documents")
    
    if stats['documents']:
        for doc in stats['documents']:
            with st.expander(doc['title'] or "Unknown Title"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**File:**", os.path.basename(doc['file_path']))
                    st.write("**Authors:**", ", ".join(doc['authors'][:3]) if doc['authors'] else "Unknown")
                    st.write("**Chunks:**", doc['text_chunks'])
                    st.write("**Sections:**", doc['sections'])
                
                with col2:
                    st.write("**Tables:**", doc['tables'])
                    st.write("**Figures:**", doc['figures'])
                    st.write("**References:**", doc['references'])
                    st.write("**Equations:**", doc['equations'])
    else:
        st.info("No documents indexed yet.")
    
    # Management actions
    st.markdown("---")
    st.subheader("🔧 Management Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Re-index Documents", type="primary"):
            with st.spinner("Re-indexing documents..."):
                try:
                    indexed_docs = agent.ingest_documents(force_reindex=True)
                    st.success(f"✅ Re-indexed {len(indexed_docs)} documents!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Re-indexing failed: {e}")
    
    with col2:
        if st.button("🗑️ Clear Conversation History"):
            agent.clear_conversation_history()
            st.success("✅ Conversation history cleared!")


def render_system_status_page(document_agent: DocumentQAAgent, arxiv_agent: ArxivAgent):
    """Render system status page."""
    st.title("⚡ System Status")
    
    # Health checks
    doc_health = document_agent.health_check()
    arxiv_health = arxiv_agent.health_check()
    
    # Overall status
    overall_status = "healthy" if doc_health['status'] == 'healthy' and arxiv_health['status'] == 'healthy' else "partial"
    
    status_color = "green" if overall_status == "healthy" else "orange"
    st.markdown(f"**Overall Status:** :{status_color}[{overall_status.upper()}]")
    
    # Document Agent Status
    st.markdown("---")
    st.subheader("📄 Document Agent")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Status:**", doc_health['status'])
        st.write("**LLM Client:**", doc_health['llm_client'])
        st.write("**Indexed Documents:**", doc_health['indexed_documents'])
    
    with col2:
        st.write("**Directories:**")
        for dir_name, exists in doc_health['directories'].items():
            status_icon = "✅" if exists else "❌"
            st.write(f"  {status_icon} {dir_name}")
    
    # Arxiv Agent Status
    st.markdown("---")
    st.subheader("📚 Arxiv Agent")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Status:**", arxiv_health['status'])
        st.write("**Arxiv API:**", arxiv_health['arxiv_api'])
    
    with col2:
        st.write("**LLM Available:**", "✅" if arxiv_health['llm_client_available'] else "❌")
        st.write("**Cache Path:**", "✅" if arxiv_health['cache_path_exists'] else "❌")
    
    # Configuration
    st.markdown("---")
    st.subheader("⚙️ Configuration")
    
    settings = config.get_all_settings()
    
    for key, value in settings.items():
        if 'key' not in key.lower():  # Don't show API keys
            st.write(f"**{key}:** {value}")


def main():
    """Main Streamlit app."""
    # Initialize agents
    document_agent, arxiv_agent = initialize_agents()
    
    if not document_agent or not arxiv_agent:
        st.error("Failed to initialize agents. Please check your configuration.")
        return
    
    # Render sidebar and get page selection
    page, settings = render_sidebar()
    
    # Render selected page
    if page == "Document Q&A":
        render_document_qa_page(document_agent, settings)
    elif page == "Arxiv Search":
        render_arxiv_page(arxiv_agent, settings)
    elif page == "Document Management":
        render_document_management_page(document_agent)
    elif page == "System Status":
        render_system_status_page(document_agent, arxiv_agent)
    
    # Footer
    st.markdown("---")
    st.markdown("🤖 **AI Document Agent** - Enterprise-ready document Q&A system")


if __name__ == "__main__":
    main()

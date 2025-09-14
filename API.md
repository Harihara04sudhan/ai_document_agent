# API Documentation - AI Document Agent

## Overview
This document provides comprehensive API documentation for the AI Document Agent system.

## Core Components

### DocumentQAAgent

The main agent class for document Q&A functionality.

#### Initialization
```python
from agents.document_agent import DocumentQAAgent

agent = DocumentQAAgent(
    llm_provider="gemini",  # Only Gemini is supported
    documents_path="./documents"
)
```

#### Methods

##### `ingest_documents(force_reindex: bool = False) -> Dict[str, str]`
Ingest and index PDF documents from the documents directory.

**Parameters:**
- `force_reindex`: Whether to force re-indexing of all documents

**Returns:**
Dictionary mapping file names to document IDs

**Example:**
```python
indexed_docs = agent.ingest_documents()
print(f"Indexed {len(indexed_docs)} documents")
```

##### `query(question: str, query_type: str = "auto", context_limit: int = 5, include_sources: bool = True) -> Dict[str, Any]`
Process a user query and return an intelligent response.

**Parameters:**
- `question`: User's question
- `query_type`: Type of query ('lookup', 'summary', 'evaluation', 'auto')
- `context_limit`: Maximum number of context chunks to use
- `include_sources`: Whether to include source information

**Returns:**
Response dictionary with answer, sources, and metadata

**Example:**
```python
response = agent.query("What are the main conclusions?")
print(response['answer'])
print(f"Confidence: {response['confidence']}")
```

##### `get_document_statistics() -> Dict[str, Any]`
Get statistics about indexed documents.

**Returns:**
Dictionary with document statistics and details

##### `health_check() -> Dict[str, Any]`
Perform a health check of the agent.

**Returns:**
Health status dictionary

### ArxivAgent

Agent for interacting with Arxiv API and retrieving research papers.

#### Initialization
```python
from agents.arxiv_agent import ArxivAgent

arxiv_agent = ArxivAgent(llm_client)
```

#### Methods

##### `search_papers(query: str, max_results: int = None, sort_by: str = "relevance", category: str = None) -> List[Dict[str, Any]]`
Search for papers on Arxiv.

**Parameters:**
- `query`: Search query
- `max_results`: Maximum number of results
- `sort_by`: Sort criteria ('relevance', 'lastUpdatedDate', 'submittedDate')
- `category`: Arxiv category to filter by

**Returns:**
List of paper information dictionaries

**Example:**
```python
papers = arxiv_agent.search_papers("neural networks", max_results=5)
for paper in papers:
    print(f"Title: {paper['title']}")
    print(f"Authors: {', '.join(paper['authors'])}")
```

##### `get_paper_details(arxiv_id: str) -> Dict[str, Any]`
Get detailed information about a specific paper.

**Parameters:**
- `arxiv_id`: Arxiv paper ID

**Returns:**
Detailed paper information

##### `recommend_papers(description: str, max_results: int = 5) -> List[Dict[str, Any]]`
Recommend papers based on a natural language description.

**Parameters:**
- `description`: Natural language description
- `max_results`: Maximum number of recommendations

**Returns:**
List of recommended papers

##### `summarize_paper(arxiv_id: str) -> Dict[str, Any]`
Generate a comprehensive summary of a paper.

**Parameters:**
- `arxiv_id`: Arxiv paper ID

**Returns:**
Paper summary with key insights

### PDFProcessor

Advanced PDF processing and content extraction.

#### Methods

##### `process_pdf(pdf_path: str) -> ExtractedContent`
Process a PDF file and extract comprehensive content.

**Parameters:**
- `pdf_path`: Path to the PDF file

**Returns:**
ExtractedContent object with all extracted information

### DocumentIndexer

Create searchable indexes from extracted content.

#### Methods

##### `index_document(file_path: str, content: ExtractedContent) -> str`
Index a document for searchability.

**Parameters:**
- `file_path`: Path to the document file
- `content`: Extracted content from the document

**Returns:**
Document ID for referencing

##### `search_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]`
Search indexed documents using semantic similarity.

**Parameters:**
- `query`: Search query
- `top_k`: Number of top results to return

**Returns:**
List of relevant document chunks with similarity scores

## Data Structures

### ExtractedContent
```python
@dataclass
class ExtractedContent:
    text: str
    title: Optional[str] = None
    authors: List[str] = None
    abstract: Optional[str] = None
    sections: List[Dict[str, str]] = None
    tables: List[Dict[str, Any]] = None
    figures: List[Dict[str, str]] = None
    references: List[str] = None
    equations: List[str] = None
    metadata: Dict[str, Any] = None
```

### Query Response Format
```python
{
    'answer': str,           # Generated answer
    'sources': List[Dict],   # Source information
    'query_type': str,       # Detected query type
    'confidence': float,     # Confidence score (0-1)
    'context_used': int,     # Number of context chunks used
    'timestamp': str         # ISO timestamp
}
```

### Paper Information Format
```python
{
    'id': str,              # Arxiv entry ID
    'arxiv_id': str,        # Short Arxiv ID
    'title': str,           # Paper title
    'authors': List[str],   # Author names
    'abstract': str,        # Paper abstract
    'published': str,       # Publication date (ISO format)
    'categories': List[str], # Arxiv categories
    'pdf_url': str,         # URL to PDF
    'entry_url': str        # URL to Arxiv page
}
```

## Configuration

### Environment Variables
```bash
# API Configuration
GEMINI_API_KEY=your_gemini_api_key
DEFAULT_LLM_PROVIDER=gemini

# Model Settings
GEMINI_MODEL=gemini-1.5-flash
MAX_TOKENS=2048
TEMPERATURE=0.3

# Processing Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Storage Configuration
VECTOR_DB_PATH=./data/vector_db
DOCUMENTS_PATH=./documents
CACHE_PATH=./data/cache

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

## Error Handling

All methods include comprehensive error handling and logging. Common exceptions:

- `ValueError`: Invalid parameters or configuration
- `FileNotFoundError`: Missing files or directories
- `ImportError`: Missing dependencies
- `ConnectionError`: API connection issues

## Rate Limiting

The system implements rate limiting for API calls:
- Gemini: Built-in rate limiting and request optimization
- Arxiv: 1 request per 3 seconds (recommended)

## Caching

- Document content is cached in `data/cache/`
- Embeddings are stored in `data/vector_db/`
- Arxiv papers are cached to reduce API calls

## Security Considerations

- API keys are managed through environment variables
- Input sanitization for all user queries
- Secure file handling for PDF processing
- No sensitive information logged

## Performance Optimization

- Parallel document processing
- Efficient vector similarity search
- Response caching
- Chunking optimization for memory usage

## Examples

### Complete Workflow Example
```python
from agents.document_agent import DocumentQAAgent
from agents.arxiv_agent import ArxivAgent

# Initialize agents
doc_agent = DocumentQAAgent()
arxiv_agent = ArxivAgent(doc_agent.llm_client)

# Ingest documents
indexed_docs = doc_agent.ingest_documents()
print(f"Indexed {len(indexed_docs)} documents")

# Query documents
response = doc_agent.query("What are the evaluation results?")
print(response['answer'])

# Search Arxiv
papers = arxiv_agent.search_papers("transformer architecture")
for paper in papers:
    print(f"{paper['title']} - {paper['authors'][0]}")

# Get paper summary
if papers:
    summary = arxiv_agent.summarize_paper(papers[0]['arxiv_id'])
    print(summary['summary'])
```

### Batch Processing Example
```python
from processors.pdf_processor import BatchPDFProcessor

processor = BatchPDFProcessor()
contents = processor.process_directory("./documents")

for filename, content in contents.items():
    print(f"Processed: {filename}")
    print(f"Title: {content.title}")
    print(f"Tables: {len(content.tables)}")
    print(f"Figures: {len(content.figures)}")
```

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

For integration testing:
```bash
python demo.py
python main.py --health
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Install dependencies with `pip install -r requirements.txt`
2. **API Key Issues**: Check `.env` file configuration
3. **PDF Processing Errors**: Ensure PDF files are not corrupted
4. **Memory Issues**: Reduce `CHUNK_SIZE` in configuration
5. **Slow Performance**: Enable caching and reduce `MAX_TOKENS`

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python main.py --query "test question"
```

### Health Checks

Regular health checks:
```bash
python main.py --health
```

This provides status of all system components.

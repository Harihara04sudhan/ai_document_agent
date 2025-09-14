# Enterprise AI Document Q&A Agent

## Overview
A lightweight AI agent that processes PDF documents and provides intelligent Q&A capabilities using Google Gemini API. The system features multi-modal content extraction, enterprise-grade optimizations, and Arxiv API integration.

## Features
- ✅ Multi-PDF document ingestion pipeline
- ✅ Multi-modal content extraction (text, tables, figures, equations)
- ✅ Intelligent Q&A interface with multiple query types
- ✅ Context-aware responses with enterprise optimizations
- ✅ Arxiv API integration for paper lookup (bonus feature)
- ✅ Secure API key management
- ✅ Comprehensive error handling and logging

## Setup Instructions

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ai_document_agent
```

2. **Create virtual environment**
```bash
python -m venv ai_document_agent_venv
source ai_document_agent_venv/bin/activate  # On Windows: ai_document_agent_venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

5. **Run the application**
```bash
python main.py
```

## Usage

### Document Ingestion
Place PDF files in the `documents/` folder and run:
```python
python ingest_documents.py
```

### Query Interface
The system supports three types of queries:

1. **Direct Content Lookup**
   - "What is the conclusion of Paper X?"
   - "What are the main findings in document Y?"

2. **Summarization**
   - "Summarize the methodology of Paper C"
   - "Give me a summary of the key insights"

3. **Evaluation Results Extraction**
   - "What are the accuracy and F1-score reported in Paper D?"
   - "Extract the performance metrics from the results section"

### Arxiv Integration (Bonus Feature)
The agent can look up papers from Arxiv:
- "Find papers about transformer architectures"
- "Look up recent papers on neural networks"

## Architecture

```
ai_document_agent/
├── main.py                 # Main application entry point
├── agents/
│   ├── __init__.py
│   ├── document_agent.py   # Core AI agent logic
│   └── arxiv_agent.py     # Arxiv API integration
├── processors/
│   ├── __init__.py
│   ├── pdf_processor.py   # PDF processing and extraction
│   └── content_extractor.py # Multi-modal content extraction
├── utils/
│   ├── __init__.py
│   ├── llm_client.py      # LLM API clients
│   └── config.py          # Configuration management
├── documents/             # Input PDF documents
├── data/                 # Processed document data
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── .env.example         # Environment variables template
```

## Technical Implementation

### Document Processing Pipeline
1. **PDF Ingestion**: Handles multiple PDF files simultaneously
2. **Content Extraction**: Uses multi-modal LLMs to extract:
   - Text content with structure preservation
   - Tables and their relationships
   - Figures and their captions
   - Mathematical equations
   - References and citations
3. **Indexing**: Creates searchable embeddings for efficient retrieval

### Enterprise Features
- **Context Management**: Maintains conversation context across queries
- **Response Optimization**: Caches frequently accessed content
- **Error Handling**: Comprehensive error handling with graceful degradation
- **Security**: Secure API key management and input validation
- **Logging**: Detailed logging for monitoring and debugging

### API Integration
- **Google Gemini**: Primary LLM for text processing and Q&A with advanced multimodal capabilities
- **Arxiv API**: Research paper lookup and retrieval

## Testing
Run the test suite:
```bash
python -m pytest tests/
```

## Performance Considerations
- Efficient caching mechanisms
- Parallel document processing
- Optimized embedding storage
- Rate limiting for API calls

## Security Features
- Environment-based API key management
- Input sanitization and validation
- Secure file handling

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License.

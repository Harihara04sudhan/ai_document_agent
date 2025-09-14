"""
Arxiv API integration for paper lookup and retrieval (Bonus Feature).
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

try:
    import arxiv
    import requests
except ImportError:
    print("arxiv library not available. Install with: pip install arxiv")
    arxiv = None
    requests = None

from utils.config import config
from utils.llm_client import BaseLLMClient


logger = logging.getLogger(__name__)


class ArxivAgent:
    """
    AI agent for interacting with Arxiv API and retrieving research papers.
    """
    
    def __init__(self, llm_client: BaseLLMClient = None):
        """
        Initialize the Arxiv Agent.
        
        Args:
            llm_client: LLM client for processing paper content
        """
        self.llm_client = llm_client
        self.max_results = config.arxiv_max_results
        self.cache_path = os.path.join(config.cache_path, "arxiv_papers")
        os.makedirs(self.cache_path, exist_ok=True)
        
        logger.info("Arxiv Agent initialized")
    
    def search_papers(
        self, 
        query: str, 
        max_results: int = None,
        sort_by: str = "relevance",
        category: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for papers on Arxiv based on query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            sort_by: Sort criteria ('relevance', 'lastUpdatedDate', 'submittedDate')
            category: Arxiv category to filter by
            
        Returns:
            List of paper information
        """
        if not arxiv:
            raise ImportError("arxiv library not installed. Install with: pip install arxiv")
        
        max_results = max_results or self.max_results
        logger.info(f"Searching Arxiv for: {query} (max_results: {max_results})")
        
        try:
            # Build search query
            search_query = query
            if category:
                search_query = f"cat:{category} AND {query}"
            
            # Configure search
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=getattr(arxiv.SortCriterion, sort_by, arxiv.SortCriterion.Relevance),
                sort_order=arxiv.SortOrder.Descending
            )
            
            # Execute search and process results
            papers = []
            for result in search.results():
                paper_info = self._process_arxiv_result(result)
                papers.append(paper_info)
            
            logger.info(f"Found {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching Arxiv: {e}")
            raise
    
    def _process_arxiv_result(self, result) -> Dict[str, Any]:
        """Process an individual Arxiv search result."""
        try:
            # Extract paper information
            paper_info = {
                'id': result.entry_id,
                'arxiv_id': result.entry_id.split('/')[-1],
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'summary': result.summary,
                'abstract': result.summary,  # Arxiv uses 'summary' for abstract
                'published': result.published.isoformat() if result.published else None,
                'updated': result.updated.isoformat() if result.updated else None,
                'categories': result.categories,
                'primary_category': result.primary_category,
                'pdf_url': result.pdf_url,
                'entry_url': result.entry_id,
                'journal_ref': result.journal_ref,
                'doi': result.doi,
                'comment': result.comment
            }
            
            return paper_info
            
        except Exception as e:
            logger.error(f"Error processing Arxiv result: {e}")
            return {}
    
    def get_paper_details(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific paper by Arxiv ID.
        
        Args:
            arxiv_id: Arxiv paper ID (e.g., '2301.12345')
            
        Returns:
            Detailed paper information
        """
        if not arxiv:
            raise ImportError("arxiv library not installed")
        
        logger.info(f"Fetching paper details for: {arxiv_id}")
        
        try:
            # Check cache first
            cache_file = os.path.join(self.cache_path, f"{arxiv_id.replace('/', '_')}.json")
            if os.path.exists(cache_file):
                logger.info(f"Loading from cache: {cache_file}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # Fetch from Arxiv
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(search.results())
            
            if not results:
                raise ValueError(f"Paper not found: {arxiv_id}")
            
            paper_info = self._process_arxiv_result(results[0])
            
            # Cache the result
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(paper_info, f, indent=2, ensure_ascii=False)
            
            return paper_info
            
        except Exception as e:
            logger.error(f"Error fetching paper details: {e}")
            raise
    
    def recommend_papers(self, description: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend papers based on a natural language description.
        
        Args:
            description: Natural language description of desired papers
            max_results: Maximum number of recommendations
            
        Returns:
            List of recommended papers
        """
        logger.info(f"Generating paper recommendations for: {description}")
        
        try:
            # Use LLM to extract search keywords from description
            search_query = self._generate_search_query(description)
            
            # Search for papers
            papers = self.search_papers(search_query, max_results)
            
            # If LLM is available, rank papers by relevance to description
            if self.llm_client and papers:
                papers = self._rank_papers_by_relevance(papers, description)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise
    
    def _generate_search_query(self, description: str) -> str:
        """Generate Arxiv search query from natural language description."""
        if not self.llm_client:
            # Simple keyword extraction
            return self._extract_keywords(description)
        
        try:
            system_message = """You are an expert at converting natural language descriptions into effective Arxiv search queries. 
Generate a concise search query that will find the most relevant papers on Arxiv."""
            
            prompt = f"""
Convert this description into an effective Arxiv search query:
"{description}"

Return only the search query, focusing on the most important technical terms and concepts.
Example output: "neural networks transformer architecture"
"""
            
            search_query = self.llm_client.generate_response(
                prompt=prompt,
                system_message=system_message,
                max_tokens=100,
                temperature=0.3
            )
            
            return search_query.strip()
            
        except Exception as e:
            logger.error(f"Error generating search query: {e}")
            return self._extract_keywords(description)
    
    def _extract_keywords(self, text: str) -> str:
        """Simple keyword extraction from text."""
        # Remove common words and extract meaningful terms
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'papers',
            'research', 'study', 'studies', 'find', 'look', 'looking', 'want'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if len(word) > 2 and word not in common_words]
        
        return ' '.join(keywords[:10])  # Limit to top 10 keywords
    
    def _rank_papers_by_relevance(self, papers: List[Dict[str, Any]], description: str) -> List[Dict[str, Any]]:
        """Rank papers by relevance to description using LLM."""
        try:
            # Create a prompt with papers for ranking
            papers_text = ""
            for i, paper in enumerate(papers):
                papers_text += f"""
Paper {i+1}:
Title: {paper['title']}
Abstract: {paper['abstract'][:300]}...
Categories: {', '.join(paper['categories'])}
---
"""
            
            system_message = """You are an expert at evaluating research paper relevance. 
Rank the papers by how well they match the user's description."""
            
            prompt = f"""
User is looking for: "{description}"

Here are the papers to rank:
{papers_text}

Please rank these papers by relevance (1 being most relevant) and provide the ranking as a simple list:
1. [Most relevant paper number]
2. [Second most relevant paper number]
etc.
"""
            
            ranking_response = self.llm_client.generate_response(
                prompt=prompt,
                system_message=system_message,
                max_tokens=200,
                temperature=0.3
            )
            
            # Parse ranking and reorder papers
            ranking = self._parse_ranking(ranking_response, len(papers))
            
            if ranking:
                # Reorder papers based on ranking
                ranked_papers = []
                for paper_idx in ranking:
                    if 0 <= paper_idx < len(papers):
                        ranked_papers.append(papers[paper_idx])
                
                # Add any missing papers
                for i, paper in enumerate(papers):
                    if i not in ranking:
                        ranked_papers.append(paper)
                
                return ranked_papers
            
            return papers
            
        except Exception as e:
            logger.error(f"Error ranking papers: {e}")
            return papers
    
    def _parse_ranking(self, ranking_text: str, num_papers: int) -> List[int]:
        """Parse ranking from LLM response."""
        try:
            # Extract numbers from ranking text
            numbers = re.findall(r'(\d+)', ranking_text)
            ranking = []
            
            for num_str in numbers:
                num = int(num_str) - 1  # Convert to 0-based indexing
                if 0 <= num < num_papers and num not in ranking:
                    ranking.append(num)
            
            return ranking
            
        except Exception as e:
            logger.error(f"Error parsing ranking: {e}")
            return []
    
    def summarize_paper(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of a paper.
        
        Args:
            arxiv_id: Arxiv paper ID
            
        Returns:
            Paper summary with key insights
        """
        if not self.llm_client:
            raise ValueError("LLM client required for paper summarization")
        
        logger.info(f"Generating summary for paper: {arxiv_id}")
        
        try:
            # Get paper details
            paper_info = self.get_paper_details(arxiv_id)
            
            # Generate summary using LLM
            system_message = """You are an expert at summarizing research papers. 
Provide comprehensive summaries that highlight the key contributions, methodology, and findings."""
            
            prompt = f"""
Summarize this research paper:

Title: {paper_info['title']}
Authors: {', '.join(paper_info['authors'])}
Abstract: {paper_info['abstract']}
Categories: {', '.join(paper_info['categories'])}

Provide a structured summary including:
1. Main contribution/objective
2. Methodology approach
3. Key findings
4. Significance/impact
5. Limitations (if apparent from abstract)
"""
            
            summary = self.llm_client.generate_response(
                prompt=prompt,
                system_message=system_message,
                max_tokens=500,
                temperature=0.4
            )
            
            return {
                'arxiv_id': arxiv_id,
                'title': paper_info['title'],
                'authors': paper_info['authors'],
                'published': paper_info['published'],
                'categories': paper_info['categories'],
                'summary': summary,
                'pdf_url': paper_info['pdf_url'],
                'arxiv_url': paper_info['entry_url'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error summarizing paper: {e}")
            raise
    
    def get_related_papers(self, arxiv_id: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find papers related to a given paper.
        
        Args:
            arxiv_id: Arxiv paper ID
            max_results: Maximum number of related papers
            
        Returns:
            List of related papers
        """
        logger.info(f"Finding related papers for: {arxiv_id}")
        
        try:
            # Get the original paper
            original_paper = self.get_paper_details(arxiv_id)
            
            # Extract key terms from title and abstract
            key_terms = self._extract_key_terms(original_paper)
            
            # Search for related papers
            search_query = ' '.join(key_terms[:5])  # Use top 5 terms
            related_papers = self.search_papers(search_query, max_results + 1)
            
            # Filter out the original paper
            related_papers = [
                paper for paper in related_papers 
                if paper['arxiv_id'] != original_paper['arxiv_id']
            ][:max_results]
            
            return related_papers
            
        except Exception as e:
            logger.error(f"Error finding related papers: {e}")
            raise
    
    def _extract_key_terms(self, paper_info: Dict[str, Any]) -> List[str]:
        """Extract key terms from paper title and abstract."""
        try:
            text = f"{paper_info['title']} {paper_info['abstract']}"
            
            if self.llm_client:
                # Use LLM to extract key terms
                system_message = "Extract the most important technical terms and concepts from this text."
                
                prompt = f"""
Extract 10 key technical terms from this research paper text:
{text}

Return only the terms, separated by commas.
Example: neural networks, transformer, attention mechanism, deep learning
"""
                
                response = self.llm_client.generate_response(
                    prompt=prompt,
                    system_message=system_message,
                    max_tokens=100,
                    temperature=0.3
                )
                
                terms = [term.strip() for term in response.split(',')]
                return [term for term in terms if len(term) > 2]
            
            else:
                # Simple keyword extraction
                return self._extract_keywords(text).split()[:10]
                
        except Exception as e:
            logger.error(f"Error extracting key terms: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of Arxiv agent."""
        try:
            # Test Arxiv API
            arxiv_status = "healthy"
            try:
                # Try a simple search
                test_search = arxiv.Search(query="test", max_results=1)
                list(test_search.results())
            except Exception as e:
                arxiv_status = f"error: {str(e)}"
            
            return {
                'status': 'healthy' if arxiv_status == 'healthy' else 'unhealthy',
                'arxiv_api': arxiv_status,
                'llm_client_available': self.llm_client is not None,
                'cache_path_exists': os.path.exists(self.cache_path),
                'max_results': self.max_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

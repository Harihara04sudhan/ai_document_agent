"""
Advanced PDF processing and content extraction module.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

try:
    import PyPDF2
    import fitz  # PyMuPDF
    import pdfplumber
except ImportError as e:
    print(f"PDF processing libraries not installed: {e}")
    print("Install with: pip install PyPDF2 PyMuPDF pdfplumber")

import re
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """Data class for extracted PDF content."""
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


class PDFProcessor:
    """Advanced PDF processor with multi-modal extraction capabilities."""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
        
    def process_pdf(self, pdf_path: str) -> ExtractedContent:
        """
        Process a PDF file and extract comprehensive content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ExtractedContent object with all extracted information
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract basic text content
            text_content = self._extract_text_pypdf2(pdf_path)
            
            # Extract structured content with PyMuPDF
            structured_content = self._extract_structured_content_pymupdf(pdf_path)
            
            # Extract tables with pdfplumber
            tables = self._extract_tables_pdfplumber(pdf_path)
            
            # Parse academic paper structure
            parsed_structure = self._parse_academic_structure(text_content)
            
            # Extract figures and captions
            figures = structured_content.get('figures', [])
            
            # Extract equations
            equations = self._extract_equations(text_content)
            
            # Get metadata
            metadata = self._extract_metadata_pypdf2(pdf_path)
            
            return ExtractedContent(
                text=text_content,
                title=parsed_structure.get('title'),
                authors=parsed_structure.get('authors', []),
                abstract=parsed_structure.get('abstract'),
                sections=parsed_structure.get('sections', []),
                tables=tables,
                figures=figures,
                references=parsed_structure.get('references', []),
                equations=equations,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def _extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract basic text using PyPDF2."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
            return ""
    
    def _extract_structured_content_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract structured content using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            structured_content = {
                'figures': [],
                'images': [],
                'fonts': [],
                'blocks': []
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract images and figures
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = {
                            'page': page_num + 1,
                            'index': img_index,
                            'width': pix.width,
                            'height': pix.height,
                            'xref': xref
                        }
                        structured_content['figures'].append(img_data)
                    pix = None
                
                # Extract text blocks with formatting
                blocks = page.get_text("dict")
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        block_text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"]
                        
                        if block_text.strip():
                            structured_content['blocks'].append({
                                'text': block_text.strip(),
                                'bbox': block.get('bbox', []),
                                'page': page_num + 1
                            })
            
            doc.close()
            return structured_content
            
        except Exception as e:
            logger.error(f"Error extracting structured content with PyMuPDF: {e}")
            return {'figures': [], 'images': [], 'fonts': [], 'blocks': []}
    
    def _extract_tables_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber."""
        try:
            tables = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 0:
                            # Process table data
                            headers = table[0] if table[0] else []
                            rows = table[1:] if len(table) > 1 else []
                            
                            table_data = {
                                'page': page_num + 1,
                                'table_index': table_idx,
                                'headers': headers,
                                'rows': rows,
                                'row_count': len(rows),
                                'column_count': len(headers),
                                'raw_data': table
                            }
                            tables.append(table_data)
            
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables with pdfplumber: {e}")
            return []
    
    def _parse_academic_structure(self, text: str) -> Dict[str, Any]:
        """Parse academic paper structure from text."""
        structure = {
            'title': None,
            'authors': [],
            'abstract': None,
            'sections': [],
            'references': []
        }
        
        try:
            lines = text.split('\n')
            
            # Extract title (usually first few non-empty lines)
            title_candidates = []
            for line in lines[:10]:
                line = line.strip()
                if line and len(line) > 10 and not line.lower().startswith(('abstract', 'introduction')):
                    title_candidates.append(line)
            
            if title_candidates:
                structure['title'] = title_candidates[0]
            
            # Extract abstract
            abstract_match = re.search(r'abstract\s*:?\s*(.*?)(?=\n\s*\n|\n\s*1\.|\n\s*introduction)', 
                                     text, re.IGNORECASE | re.DOTALL)
            if abstract_match:
                structure['abstract'] = abstract_match.group(1).strip()
            
            # Extract sections
            section_pattern = r'\n\s*(\d+\.?\s*[A-Z][^.\n]*)\n'
            sections = re.findall(section_pattern, text)
            structure['sections'] = [{'title': section.strip(), 'content': ''} for section in sections]
            
            # Extract references
            ref_start = text.lower().find('references')
            if ref_start != -1:
                references_text = text[ref_start:]
                ref_lines = [line.strip() for line in references_text.split('\n') 
                           if line.strip() and not line.strip().lower() == 'references']
                structure['references'] = ref_lines[:50]  # Limit to 50 references
            
            return structure
            
        except Exception as e:
            logger.error(f"Error parsing academic structure: {e}")
            return structure
    
    def _extract_equations(self, text: str) -> List[str]:
        """Extract mathematical equations from text."""
        try:
            # Look for common equation patterns
            equation_patterns = [
                r'\$\$([^$]+)\$\$',  # LaTeX display math
                r'\$([^$]+)\$',      # LaTeX inline math
                r'\\begin\{equation\}(.*?)\\end\{equation\}',  # LaTeX equations
                r'\\begin\{align\}(.*?)\\end\{align\}',        # LaTeX align
            ]
            
            equations = []
            for pattern in equation_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                equations.extend([match.strip() for match in matches])
            
            return equations[:20]  # Limit to 20 equations
            
        except Exception as e:
            logger.error(f"Error extracting equations: {e}")
            return []
    
    def _extract_metadata_pypdf2(self, pdf_path: str) -> Dict[str, Any]:
        """Extract PDF metadata using PyPDF2."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = {}
                
                if pdf_reader.metadata:
                    for key, value in pdf_reader.metadata.items():
                        metadata[key] = value
                
                metadata['page_count'] = len(pdf_reader.pages)
                metadata['file_size'] = os.path.getsize(pdf_path)
                metadata['file_name'] = os.path.basename(pdf_path)
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}


class BatchPDFProcessor:
    """Process multiple PDFs in batch."""
    
    def __init__(self):
        self.processor = PDFProcessor()
    
    def process_directory(self, directory_path: str) -> Dict[str, ExtractedContent]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            Dictionary mapping file names to extracted content
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        results = {}
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            try:
                logger.info(f"Processing: {pdf_file}")
                content = self.processor.process_pdf(pdf_path)
                results[pdf_file] = content
                logger.info(f"Successfully processed: {pdf_file}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                continue
        
        return results
    
    def save_extracted_content(self, content: ExtractedContent, output_path: str) -> None:
        """Save extracted content to JSON file."""
        try:
            # Convert to serializable dictionary
            content_dict = {
                'text': content.text,
                'title': content.title,
                'authors': content.authors or [],
                'abstract': content.abstract,
                'sections': content.sections or [],
                'tables': content.tables or [],
                'figures': content.figures or [],
                'references': content.references or [],
                'equations': content.equations or [],
                'metadata': content.metadata or {}
            }
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(content_dict, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved extracted content to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving content to {output_path}: {e}")
            raise

"""
Earnings Calls Service - Core functionality for fetching, parsing, and analyzing earnings calls
"""

import os
import re
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import requests
import chromadb
from bs4 import BeautifulSoup
import pdfplumber
from io import BytesIO
import json
from .openai_earnings_service import OpenAIEarningsService
from .metric_extractor import FinancialMetricExtractor, MetricExtractionResult

logger = logging.getLogger(__name__)

@dataclass
class EarningsDocument:
    """Represents an earnings call document"""
    doc_id: str
    ticker: str
    quarter: str
    artifact: str  # earnings_call_transcript, press_release, etc.
    source_url: str
    content: str
    page_anchors: Dict[int, str]  # page number -> anchor text
    status: str = "indexed"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class EarningsSummary:
    """Represents a structured earnings summary"""
    doc_id: str
    tldr: List[str]
    guidance: Dict[str, Any]
    kpis: List[Dict[str, Any]]
    risks: List[str]
    quotes: List[Dict[str, str]]  # quote, speaker, page
    sources: List[str]

class PerplexityAPI:
    """Perplexity API client for source discovery"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def search_earnings_sources(self, ticker: str, quarter: str) -> List[Dict[str, str]]:
        """Search for earnings call sources using multiple methods"""
        sources = []
        
        # Method 1: Try Yahoo Finance earnings calls page
        yahoo_sources = self._get_yahoo_finance_sources(ticker, quarter)
        if yahoo_sources:
            sources.extend(yahoo_sources)
        
        # Method 2: Try Perplexity API for additional sources
        perplexity_sources = self._get_perplexity_sources(ticker, quarter)
        if perplexity_sources:
            sources.extend(perplexity_sources)
        
        # Method 3: Try direct company IR page patterns
        ir_sources = self._get_company_ir_sources(ticker, quarter)
        if ir_sources:
            sources.extend(ir_sources)
        
        if sources:
            logger.info(f"Found {len(sources)} sources for {ticker} {quarter}: {sources}")
        else:
            logger.warning(f"No sources found for {ticker} {quarter}")
        
        return sources
    
    def _get_yahoo_finance_sources(self, ticker: str, quarter: str) -> List[Dict[str, str]]:
        """Get sources from Yahoo Finance earnings calls page"""
        try:
            yahoo_url = f"https://finance.yahoo.com/quote/{ticker}/earnings-calls/"
            logger.info(f"Checking Yahoo Finance: {yahoo_url}")
            
            # This would require parsing the Yahoo Finance page
            # For now, return the URL as a potential source
            return [{"yahoo_earnings": yahoo_url}]
            
        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
            return []
    
    def _get_perplexity_sources(self, ticker: str, quarter: str) -> List[Dict[str, str]]:
        """Get sources using Perplexity API"""
        try:
            query = f"{ticker} {quarter} earnings call transcript PDF investor relations"
            
            payload = {
                "model": "sonar-pro",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Find the official investor relations page and earnings call transcript PDF for {ticker} {quarter}. Return only the direct URLs to the IR page and transcript PDF if available. Format as JSON with keys: ir_page, transcript_pdf, press_release"
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Try to extract URLs from the response
            urls = self._extract_urls(content)
            return urls if urls else []
            
        except Exception as e:
            logger.error(f"Perplexity API error: {e}")
            return []
    
    def _get_company_ir_sources(self, ticker: str, quarter: str) -> List[Dict[str, str]]:
        """Get sources from common company IR page patterns"""
        try:
            # Company-specific IR page patterns
            company_ir_patterns = {
                'AAPL': [
                    'https://investor.apple.com/',
                    'https://investor.apple.com/investor-relations/',
                    'https://investor.apple.com/earnings/',
                ],
                'MSFT': [
                    'https://www.microsoft.com/en-us/investor/',
                    'https://www.microsoft.com/en-us/investor/earnings/',
                ],
                'GOOGL': [
                    'https://abc.xyz/investor/',
                    'https://abc.xyz/investor/earnings/',
                ],
                'AMZN': [
                    'https://ir.aboutamazon.com/',
                    'https://ir.aboutamazon.com/earnings/',
                ],
                'META': [
                    'https://investor.fb.com/',
                    'https://investor.fb.com/earnings/',
                ],
                'TSLA': [
                    'https://ir.tesla.com/',
                    'https://ir.tesla.com/earnings/',
                ],
                'NVDA': [
                    'https://investor.nvidia.com/',
                    'https://investor.nvidia.com/earnings/',
                ],
            }
            
            # Get company-specific patterns or use generic ones
            patterns = company_ir_patterns.get(ticker.upper(), [
                f"https://investor.{ticker.lower()}.com/",
                f"https://{ticker.lower()}.com/investor-relations/",
            ])
            
            sources = []
            for pattern in patterns:
                sources.append({"ir_page": pattern})
            
            return sources
            
        except Exception as e:
            logger.error(f"Company IR sources error: {e}")
            return []
    
    
    def _extract_urls(self, content: str) -> List[Dict[str, str]]:
        """Extract URLs from Perplexity response"""
        urls = []
        
        # Look for JSON-like structure
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                # Convert to list format if it's a dict
                if isinstance(data, dict):
                    return [data]
                return data
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")
        
        # Fallback: extract URLs using regex
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        found_urls = re.findall(url_pattern, content)
        
        # Categorize URLs
        for url in found_urls:
            if 'ir.' in url.lower() or 'investor' in url.lower():
                urls.append({'ir_page': url})
            elif 'transcript' in url.lower() or 'pdf' in url.lower():
                urls.append({'transcript_pdf': url})
            elif 'press' in url.lower() or 'release' in url.lower():
                urls.append({'press_release': url})
        
        return urls

class PDFParser:
    """PDF parsing utilities for earnings transcripts"""
    
    @staticmethod
    def parse_pdf_from_url(url: str) -> Tuple[str, Dict[int, str]]:
        """Parse PDF from URL and extract text with page anchors"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/pdf,application/octet-stream,*/*;q=0.9',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                text_content = []
                page_anchors = {}
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                        # Create page anchor (first few words)
                        first_words = ' '.join(page_text.split()[:10])
                        page_anchors[page_num] = first_words
                
                full_text = '\n\n'.join(text_content)
                return full_text, page_anchors
                
        except Exception as e:
            logger.error(f"PDF parsing error: {e}")
            return "", {}
    
    
    @staticmethod
    def parse_html_from_url(url: str) -> Tuple[str, Dict[int, str]]:
        """Parse HTML from URL and extract text with section anchors"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text_content = soup.get_text()
            page_anchors = {1: "HTML Content"}
            
            return text_content, page_anchors
            
        except Exception as e:
            logger.error(f"HTML parsing error: {e}")
            return "", {}

class VectorDatabase:
    """ChromaDB wrapper for document storage and retrieval"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Try to get existing collection first
        try:
            self.collection = self.client.get_collection("earnings_documents")
            # Test if collection has correct dimensions by trying to add a test embedding
            test_embedding = [0.0] * 1536  # OpenAI embedding dimension
            try:
                self.collection.add(
                    ids=["test_dimension_check"],
                    embeddings=[test_embedding],
                    documents=["test"],
                    metadatas=[{"test": True}]
                )
                # If successful, remove the test document
                self.collection.delete(ids=["test_dimension_check"])
            except Exception as e:
                # Collection has wrong dimensions, recreate it
                logger.info("Collection has wrong dimensions, recreating...")
                self.client.delete_collection("earnings_documents")
                self.collection = self.client.create_collection(
                    name="earnings_documents",
                    metadata={"hnsw:space": "cosine"}
                )
        except:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name="earnings_documents",
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_document(self, doc: EarningsDocument) -> bool:
        """Add document to vector database"""
        try:
            # Chunk the document
            chunks = self._chunk_document(doc)
            
            # Prepare data for ChromaDB
            chunk_ids = [f"{doc.doc_id}_{i}" for i in range(len(chunks))]
            embeddings = [self._get_embedding(chunk) for chunk in chunks]
            metadatas = [{
                "ticker": doc.ticker,
                "quarter": doc.quarter,
                "artifact": doc.artifact,
                "source_url": doc.source_url,
                "page": i + 1,
                "created_at": doc.created_at.isoformat()
            } for i in range(len(chunks))]
            
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to vector DB: {e}")
            return False
    
    def search_documents(self, query: str, ticker: str = None, quarter: str = None, 
                        n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks"""
        try:
            query_embedding = self._get_embedding(query)
            
            where_clause = None
            if ticker and quarter:
                where_clause = {"$and": [{"ticker": ticker}, {"quarter": quarter}]}
            elif ticker:
                where_clause = {"ticker": ticker}
            elif quarter:
                where_clause = {"quarter": quarter}
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector DB: {e}")
            return []
    
    def _chunk_document(self, doc: EarningsDocument, chunk_size: int = 1000) -> List[str]:
        """Split document into chunks for embedding"""
        text = doc.content
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI embeddings"""
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            # Fallback to hash-based embedding if OpenAI fails
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            return [float(b) / 255.0 for b in hash_bytes[:16]] + [0.0] * (1536 - 16)

class EarningsService:
    """Main earnings service orchestrator"""
    
    def __init__(self):
        self.perplexity = PerplexityAPI(os.getenv('PERPLEXITY_API_KEY'))
        self.pdf_parser = PDFParser()
        self.vector_db = VectorDatabase()
        self.openai_service = OpenAIEarningsService()
        self.metric_extractor = FinancialMetricExtractor()
    
    def process_earnings_query(self, query: str) -> Dict[str, Any]:
        """Main entry point for processing earnings queries"""
        try:
            # Parse query to extract ticker and quarter
            ticker, quarter = self._parse_earnings_query(query)
            
            if not ticker or not quarter:
                return {"error": "Could not extract ticker and quarter from query"}
            
            # Check if we already have this document
            doc_id = self._generate_doc_id(ticker, quarter)
            existing_doc = self._get_existing_document(doc_id)
            
            if existing_doc:
                return self._generate_summary_response(existing_doc)
            
            # Discover sources
            sources = self.perplexity.search_earnings_sources(ticker, quarter)
            
            if not sources:
                return {"error": "Could not find earnings call sources"}
            
            # Fetch and parse document
            doc = self._fetch_and_parse_document(ticker, quarter, sources)
            
            if not doc:
                # No real sources found - return error instead of mock data
                logger.warning(f"No real sources found for {ticker} {quarter}")
                return {
                    "error": f"Could not find earnings call sources for {ticker} {quarter}. Please check if the earnings call is available or try a different quarter.",
                    "success": False
                }
            
            # Store in vector database
            self.vector_db.add_document(doc)
            
            # Generate summary
            summary = self._generate_earnings_summary(doc)
            
            return self._format_summary_response(doc, summary)
            
        except Exception as e:
            logger.error(f"Error processing earnings query: {e}")
            return {"error": str(e)}
    
    def _parse_earnings_query(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse query to extract ticker and quarter"""
        query_lower = query.lower()
        
        # Extract ticker using the same logic as intelligent analyzer
        ticker = self._extract_ticker_from_query(query)
        
        # Extract quarter
        quarter_patterns = [
            r'q([1-4])\s*(\d{4})',  # Q1 2024, Q2 2025, etc.
            r'(\d{4})\s*q([1-4])',  # 2024 Q1, 2025 Q2, etc.
            r'quarter\s*([1-4])\s*(\d{4})',  # quarter 1 2024
        ]
        
        quarter = None
        for pattern in quarter_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if len(match.groups()) == 2:
                    q, year = match.groups()
                    quarter = f"{year}Q{q}"
                break
        
        return ticker, quarter
    
    def _extract_ticker_from_query(self, query: str) -> Optional[str]:
        """Extract ticker symbol from query using company name mapping"""
        import re
        
        # Look for $TICKER format
        dollar_match = re.search(r'\$([A-Z]{1,5})\b', query.upper())
        if dollar_match:
            return dollar_match.group(1)
        
        # Company name to ticker mapping
        company_mapping = {
            'apple': 'AAPL',
            'microsoft': 'MSFT', 
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'amazon': 'AMZN',
            'nvidia': 'NVDA',
            'tesla': 'TSLA',
            'meta': 'META',
            'facebook': 'META',
            'netflix': 'NFLX',
            'shopify': 'SHOP',
            'uber': 'UBER',
            'airbnb': 'ABNB',
            'spotify': 'SPOT',
            'zoom': 'ZM',
            'salesforce': 'CRM',
            'oracle': 'ORCL',
            'adobe': 'ADBE',
            'intel': 'INTC',
            'cisco': 'CSCO',
            'ibm': 'IBM',
            'nike': 'NKE',
            'mcdonalds': 'MCD',
            'coca cola': 'KO',
            'pepsi': 'PEP',
            'walmart': 'WMT',
            'target': 'TGT',
            'home depot': 'HD',
            'lowes': 'LOW',
            'boeing': 'BA',
            'general motors': 'GM',
            'ford': 'F',
            'disney': 'DIS',
            'comcast': 'CMCSA',
            'verizon': 'VZ',
            'at&t': 'T',
            't mobile': 'TMUS',
            'sprint': 'S',
            'paypal': 'PYPL',
            'square': 'SQ',
            'stripe': 'STRIPE',
            'coinbase': 'COIN',
            'robinhood': 'HOOD',
            'palantir': 'PLTR',
            'snowflake': 'SNOW',
            'databricks': 'DATABRICKS',
            'openai': 'OPENAI',
            'anthropic': 'ANTHROPIC'
        }
        
        query_lower = query.lower()
        
        # Check for company names
        for company, ticker in company_mapping.items():
            if company in query_lower:
                return ticker
        
        # Look for common tickers
        common_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'SHOP']
        for ticker in common_tickers:
            if ticker in query.upper():
                return ticker
        
        return None
    
    def _generate_doc_id(self, ticker: str, quarter: str) -> str:
        """Generate unique document ID"""
        content = f"{ticker}_{quarter}_earnings_call"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _get_existing_document(self, doc_id: str) -> Optional[EarningsDocument]:
        """Check if document already exists in vector DB"""
        # This would query the vector DB for existing documents
        # For now, return None to always fetch new documents
        return None
    
    def _fetch_and_parse_document(self, ticker: str, quarter: str, sources: List[Dict[str, str]]) -> Optional[EarningsDocument]:
        """Fetch and parse earnings document from sources"""
        for source in sources:
            # Try transcript PDF first
            if 'transcript_pdf' in source and source['transcript_pdf'] and source['transcript_pdf'] != 'null':
                url = source['transcript_pdf']
                content, page_anchors = self.pdf_parser.parse_pdf_from_url(url)
                if content and self._is_valid_earnings_content(content):
                    doc_id = self._generate_doc_id(ticker, quarter)
                    return EarningsDocument(
                        doc_id=doc_id,
                        ticker=ticker,
                        quarter=quarter,
                        artifact="earnings_call_transcript",
                        source_url=url,
                        content=content,
                        page_anchors=page_anchors
                    )
            
            # Try Yahoo Finance earnings calls page
            if 'yahoo_earnings' in source and source['yahoo_earnings'] and source['yahoo_earnings'] != 'null':
                url = source['yahoo_earnings']
                content, page_anchors = self.pdf_parser.parse_html_from_url(url)
                if content and self._is_valid_earnings_content(content):
                    doc_id = self._generate_doc_id(ticker, quarter)
                    return EarningsDocument(
                        doc_id=doc_id,
                        ticker=ticker,
                        quarter=quarter,
                        artifact="yahoo_earnings_page",
                        source_url=url,
                        content=content,
                        page_anchors=page_anchors
                    )
            
            # Try IR page as fallback
            if 'ir_page' in source and source['ir_page'] and source['ir_page'] != 'null':
                url = source['ir_page']
                content, page_anchors = self.pdf_parser.parse_html_from_url(url)
                if content and self._is_valid_earnings_content(content):
                    doc_id = self._generate_doc_id(ticker, quarter)
                    return EarningsDocument(
                        doc_id=doc_id,
                        ticker=ticker,
                        quarter=quarter,
                        artifact="earnings_call_transcript",
                        source_url=url,
                        content=content,
                        page_anchors=page_anchors
                    )
            
            # Try press release as final fallback
            if 'press_release' in source and source['press_release'] and source['press_release'] != 'null':
                url = source['press_release']
                content, page_anchors = self.pdf_parser.parse_html_from_url(url)
                if content and self._is_valid_earnings_content(content):
                    doc_id = self._generate_doc_id(ticker, quarter)
                    return EarningsDocument(
                        doc_id=doc_id,
                        ticker=ticker,
                        quarter=quarter,
                        artifact="earnings_press_release",
                        source_url=url,
                        content=content,
                        page_anchors=page_anchors
                    )
        
        # If no real sources work, return None to trigger mock fallback
        return None
    
    def _is_valid_earnings_content(self, content: str) -> bool:
        """Validate that content actually contains earnings call data"""
        content_lower = content.lower()
        
        # Check for blocked/error messages (more specific patterns)
        blocked_patterns = [
            'your request has been blocked',
            'access denied',
            'forbidden',
            'this page is blocked',
            'robot detection',
            'captcha verification',
            'cloudflare protection',
            'unauthorized access to this resource',
            'blocked by administrator'
        ]
        
        for pattern in blocked_patterns:
            if pattern in content_lower:
                logger.warning(f"Content appears to be blocked: {pattern}")
                return False
        
        # Check for earnings-related content
        earnings_indicators = [
            'earnings call',
            'quarterly results',
            'revenue',
            'eps',
            'guidance',
            'conference call',
            'transcript',
            'financial results',
            'quarterly earnings',
            'q1', 'q2', 'q3', 'q4'
        ]
        
        earnings_count = sum(1 for indicator in earnings_indicators if indicator in content_lower)
        
        # Content must have at least 3 earnings-related terms and be substantial
        return earnings_count >= 3 and len(content) > 1000
    
    
    def _generate_earnings_summary(self, doc: EarningsDocument) -> EarningsSummary:
        """Generate structured summary using OpenAI LLM with metric extraction"""
        try:
            # Extract financial metrics first
            logger.info(f"Extracting financial metrics for {doc.ticker} {doc.quarter}")
            metric_result = self.metric_extractor.extract_metrics(
                doc.content, doc.ticker, doc.quarter
            )
            
            # Use OpenAI to generate real summary
            openai_summary = self.openai_service.generate_earnings_summary(
                doc.content, doc.ticker, doc.quarter
            )
            
            # Add extracted metrics to the summary
            if metric_result:
                # Format metrics for inclusion in summary
                formatted_metrics = self.metric_extractor.format_metrics_for_display(metric_result)
                metric_insights = self.metric_extractor.generate_metric_insights(metric_result, doc.ticker)
                
                # Add to KPIs if not already present
                if 'kpis' not in openai_summary:
                    openai_summary['kpis'] = []
                
                # Add extracted metrics as additional KPIs (avoid duplicates with OpenAI KPIs)
                existing_metrics = set()
                if 'kpis' in openai_summary:
                    for kpi in openai_summary['kpis']:
                        if isinstance(kpi, dict):
                            existing_metrics.add(kpi.get('metric', '').lower())
                
                for metric in formatted_metrics.get('basic_metrics', []):
                    metric_name = metric['name']
                    if metric_name.lower() not in existing_metrics:
                        # Fix unit display for EPS
                        if 'EPS' in metric_name and metric['value'].endswith('B'):
                            value = metric['value'].replace('B', '')
                            value = f"${value}"
                        else:
                            value = metric['value']
                            
                        openai_summary['kpis'].append({
                            'metric': metric_name,
                            'value': value,
                            'change': metric['change'],
                            'source': 'extracted'
                        })
                
                # Add metric insights to the summary
                if 'additional_insights' not in openai_summary:
                    openai_summary['additional_insights'] = []
                openai_summary['additional_insights'].append(metric_insights)
            
            # Convert OpenAI response to EarningsSummary format
            return EarningsSummary(
                doc_id=doc.doc_id,
                tldr=openai_summary.get('tldr', []),
                guidance=openai_summary.get('guidance', {}),
                kpis=openai_summary.get('kpis', []),
                risks=openai_summary.get('risks', []),
                quotes=openai_summary.get('quotes', []),
                sources=[doc.source_url]
            )
            
        except Exception as e:
            logger.error(f"Error generating OpenAI summary: {e}")
            # Return error if OpenAI fails
            return EarningsSummary(
                doc_id=doc.doc_id,
                tldr=[f"Error generating summary for {doc.ticker} {doc.quarter}"],
                guidance={},
                kpis=[],
                risks=[],
                quotes=[],
                sources=[doc.source_url]
            )
    
    
    def _format_summary_response(self, doc: EarningsDocument, summary: EarningsSummary) -> Dict[str, Any]:
        """Format the response for the API"""
        return {
            "doc_id": doc.doc_id,
            "ticker": doc.ticker,
            "quarter": doc.quarter,
            "tldr": summary.tldr,
            "guidance": summary.guidance,
            "kpis": summary.kpis,
            "risks": summary.risks,
            "quotes": summary.quotes,
            "sources": summary.sources,
            "success": True
        }
    
    def _generate_summary_response(self, doc: EarningsDocument) -> Dict[str, Any]:
        """Generate response for existing document"""
        # This would retrieve the cached summary
        return {
            "doc_id": doc.doc_id,
            "ticker": doc.ticker,
            "quarter": doc.quarter,
            "cached": True,
            "success": True
        }
    
    def answer_earnings_question(self, doc_id: str, question: str) -> Dict[str, Any]:
        """Answer questions about a specific earnings document using ChromaDB retrieval"""
        try:
            # Extract ticker and quarter from doc_id if possible
            ticker = None
            quarter = None
            
            # Try to extract from doc_id format: {ticker}_{quarter}_{timestamp}
            if '_' in doc_id:
                parts = doc_id.split('_')
                if len(parts) >= 2:
                    ticker = parts[0].upper()
                    quarter = parts[1].upper()
            
            # Search for relevant document chunks using ChromaDB
            search_results = self.vector_db.search_documents(
                query=question,
                ticker=ticker,
                quarter=quarter,
                n_results=5
            )
            
            if not search_results or not search_results.get('documents'):
                return {
                    "doc_id": doc_id,
                    "question": question,
                    "answer": "I couldn't find any relevant information in the earnings documents to answer your question.",
                    "citations": [],
                    "confidence": "low",
                    "success": False
                }
            
            # Combine relevant chunks for context
            relevant_chunks = search_results['documents'][0]  # ChromaDB returns list of lists
            metadatas = search_results['metadatas'][0] if search_results.get('metadatas') else []
            
            # Build context from retrieved chunks
            context = "\n\n".join(relevant_chunks)
            
            # Extract ticker and quarter from metadata if available
            if metadatas and len(metadatas) > 0:
                ticker = metadatas[0].get('ticker', ticker)
                quarter = metadatas[0].get('quarter', quarter)
            
            # Use OpenAI to answer the question with real context
            answer_data = self.openai_service.answer_earnings_question(
                question, context, ticker or "Unknown", quarter or "Unknown"
            )
            
            # Build citations from metadata
            citations = []
            if metadatas:
                for i, metadata in enumerate(metadatas):
                    page = metadata.get('page', i + 1)
                    ticker = metadata.get('ticker', 'Unknown')
                    quarter = metadata.get('quarter', 'Unknown')
                    citations.append(f"Page {page} from {ticker} {quarter} earnings call")
            
            return {
                "doc_id": doc_id,
                "question": question,
                "answer": answer_data.get("answer", "Unable to answer the question."),
                "citations": citations,
                "confidence": answer_data.get("confidence", "medium"),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error answering earnings question: {e}")
            return {
                "doc_id": doc_id,
                "question": question,
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "citations": [],
                "confidence": "low",
                "success": False
            }

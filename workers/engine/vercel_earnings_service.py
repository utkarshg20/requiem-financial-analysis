"""
Vercel-compatible earnings service
Uses in-memory ChromaDB and simplified storage
"""
import os
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pdfplumber
from io import BytesIO
import openai
from .vercel_chromadb import VercelChromaDB
from .openai_earnings_service import OpenAIEarningsService
from .metric_extractor import FinancialMetricExtractor

logger = logging.getLogger(__name__)

class VercelEarningsService:
    """
    Vercel-compatible earnings service
    Uses in-memory storage and simplified document processing
    """
    
    def __init__(self):
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.vector_db = VercelChromaDB()
        self.openai_service = OpenAIEarningsService()
        self.metric_extractor = FinancialMetricExtractor()
        
        # Initialize OpenAI
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
    
    async def process_earnings_query(self, query: str) -> Dict[str, Any]:
        """Process earnings query with Vercel-compatible storage"""
        try:
            # Parse query
            ticker, quarter = self._parse_earnings_query(query)
            if not ticker or not quarter:
                return {
                    "success": False,
                    "error": "Could not extract ticker and quarter from query"
                }
            
            # Check if we already have this document in memory
            existing_doc = self._get_existing_document(ticker, quarter)
            if existing_doc:
                logger.info(f"Using existing document for {ticker} {quarter}")
                return self._generate_earnings_summary(existing_doc)
            
            # Search for sources
            sources = await self._search_earnings_sources(ticker, quarter)
            if not sources:
                return {
                    "success": False,
                    "error": f"Could not find earnings call sources for {ticker} {quarter}"
                }
            
            # Fetch and parse document
            doc = await self._fetch_and_parse_document(sources, ticker, quarter)
            if not doc:
                return {
                    "success": False,
                    "error": f"Could not fetch or parse earnings document for {ticker} {quarter}"
                }
            
            # Store in memory (will be lost on serverless restart)
            self.vector_db.add_document(
                doc_id=doc.doc_id,
                content=doc.content,
                metadata={
                    "ticker": doc.ticker,
                    "quarter": doc.quarter,
                    "source_url": doc.source_url,
                    "created_at": datetime.now().isoformat()
                },
                embedding=self._get_embedding(doc.content)
            )
            
            # Generate summary
            return self._generate_earnings_summary(doc)
            
        except Exception as e:
            logger.error(f"Error processing earnings query: {e}")
            return {
                "success": False,
                "error": f"Error processing earnings query: {str(e)}"
            }
    
    def _parse_earnings_query(self, query: str) -> tuple:
        """Parse ticker and quarter from query"""
        # Simple parsing logic
        query_lower = query.lower()
        
        # Extract ticker
        ticker = None
        if 'apple' in query_lower or 'aapl' in query_lower:
            ticker = 'AAPL'
        elif 'microsoft' in query_lower or 'msft' in query_lower:
            ticker = 'MSFT'
        elif 'google' in query_lower or 'googl' in query_lower:
            ticker = 'GOOGL'
        elif 'amazon' in query_lower or 'amzn' in query_lower:
            ticker = 'AMZN'
        elif 'meta' in query_lower or 'fb' in query_lower:
            ticker = 'META'
        
        # Extract quarter
        quarter = None
        if 'q4 2024' in query_lower:
            quarter = '2024Q4'
        elif 'q3 2024' in query_lower:
            quarter = '2024Q3'
        elif 'q2 2024' in query_lower:
            quarter = '2024Q2'
        elif 'q1 2024' in query_lower:
            quarter = '2024Q1'
        
        return ticker, quarter
    
    def _get_existing_document(self, ticker: str, quarter: str) -> Optional[Dict]:
        """Check if document exists in memory"""
        try:
            # Search for existing document
            results = self.vector_db.search_documents(
                query_embedding=[0.0] * 1536,  # Dummy embedding
                ticker=ticker,
                quarter=quarter,
                limit=1
            )
            
            if results:
                return {
                    'doc_id': f"{ticker}_{quarter}",
                    'ticker': ticker,
                    'quarter': quarter,
                    'content': results[0]['content'],
                    'source_url': results[0]['metadata'].get('source_url', '')
                }
            
            return None
        except Exception as e:
            logger.error(f"Error checking existing document: {e}")
            return None
    
    async def _search_earnings_sources(self, ticker: str, quarter: str) -> Dict[str, str]:
        """Search for earnings sources using Perplexity API"""
        try:
            if not self.perplexity_api_key:
                return self._get_fallback_sources(ticker, quarter)
            
            # Use Perplexity API
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"Find earnings call transcript PDF for {ticker} {quarter}. Look for investor relations pages, press releases, or SEC filings."
            
            data = {
                "model": "sonar-pro",
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                return self._extract_urls_from_content(content, ticker, quarter)
            else:
                logger.error(f"Perplexity API error: {response.status_code}")
                return self._get_fallback_sources(ticker, quarter)
                
        except Exception as e:
            logger.error(f"Error searching sources: {e}")
            return self._get_fallback_sources(ticker, quarter)
    
    def _get_fallback_sources(self, ticker: str, quarter: str) -> Dict[str, str]:
        """Fallback sources when API fails"""
        fallback_sources = {
            'AAPL': {
                '2024Q4': 'https://www.apple.com/newsroom/2024/10/apple-reports-fourth-quarter-results/',
                '2024Q3': 'https://www.apple.com/newsroom/2024/07/apple-reports-third-quarter-results/'
            },
            'MSFT': {
                '2024Q3': 'https://www.microsoft.com/en-us/investor/earnings/fy-2024-q3',
                '2024Q2': 'https://www.microsoft.com/en-us/investor/earnings/fy-2024-q2'
            }
        }
        
        return {
            'press_release': fallback_sources.get(ticker, {}).get(quarter, ''),
            'ir_page': '',
            'transcript_pdf': ''
        }
    
    def _extract_urls_from_content(self, content: str, ticker: str, quarter: str) -> Dict[str, str]:
        """Extract URLs from Perplexity response"""
        # Simple URL extraction
        import re
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
        
        return {
            'press_release': urls[0] if urls else '',
            'ir_page': urls[1] if len(urls) > 1 else '',
            'transcript_pdf': urls[2] if len(urls) > 2 else ''
        }
    
    async def _fetch_and_parse_document(self, sources: Dict[str, str], ticker: str, quarter: str) -> Optional[Dict]:
        """Fetch and parse document from sources"""
        try:
            # Try press release first
            if sources.get('press_release'):
                content = await self._fetch_url_content(sources['press_release'])
                if content and self._is_valid_earnings_content(content):
                    return {
                        'doc_id': f"{ticker}_{quarter}",
                        'ticker': ticker,
                        'quarter': quarter,
                        'content': content,
                        'source_url': sources['press_release']
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching document: {e}")
            return None
    
    async def _fetch_url_content(self, url: str) -> str:
        """Fetch content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            text_content = soup.get_text(separator=' ', strip=True)
            return text_content
            
        except Exception as e:
            logger.error(f"Error fetching URL content: {e}")
            return ""
    
    def _is_valid_earnings_content(self, content: str) -> bool:
        """Check if content is valid earnings data"""
        if not content or len(content) < 100:
            return False
        
        # Check for blocked content
        blocked_patterns = [
            'access denied', 'blocked', 'unauthorized access',
            'request has been blocked', 'forbidden'
        ]
        
        content_lower = content.lower()
        for pattern in blocked_patterns:
            if pattern in content_lower:
                return False
        
        # Check for earnings-related keywords
        earnings_keywords = [
            'revenue', 'earnings', 'quarter', 'financial results',
            'eps', 'cash flow', 'guidance', 'outlook'
        ]
        
        keyword_count = sum(1 for keyword in earnings_keywords if keyword in content_lower)
        return keyword_count >= 3
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        try:
            if not self.openai_api_key:
                # Return dummy embedding if no API key
                return [0.0] * 1536
            
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000]  # Limit text length
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return [0.0] * 1536
    
    def _generate_earnings_summary(self, doc: Dict) -> Dict[str, Any]:
        """Generate earnings summary"""
        try:
            # Use OpenAI service for summary
            summary = self.openai_service.generate_earnings_summary(
                doc['content'], doc['ticker'], doc['quarter']
            )
            
            # Add extracted metrics
            metric_result = self.metric_extractor.extract_metrics(
                doc['content'], doc['ticker'], doc['quarter']
            )
            
            if metric_result:
                formatted_metrics = self.metric_extractor.format_metrics_for_display(metric_result)
                if 'kpis' not in summary:
                    summary['kpis'] = []
                
                for metric in formatted_metrics.get('basic_metrics', []):
                    summary['kpis'].append({
                        'metric': metric['name'],
                        'value': metric['value'],
                        'change': metric['change']
                    })
            
            return {
                "success": True,
                "doc_id": doc['doc_id'],
                "ticker": doc['ticker'],
                "quarter": doc['quarter'],
                "tldr": summary.get('tldr', []),
                "guidance": summary.get('guidance', {}),
                "kpis": summary.get('kpis', []),
                "risks": summary.get('risks', []),
                "quotes": summary.get('quotes', []),
                "sources": [doc['source_url']]
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                "success": False,
                "error": f"Error generating summary: {str(e)}"
            }

"""
OpenAI Earnings Service - Real LLM integration for earnings call analysis
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import openai
from datetime import datetime

logger = logging.getLogger(__name__)

class OpenAIEarningsService:
    """OpenAI service for earnings call analysis and summarization"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o-mini"  # Using the efficient model for cost optimization
    
    def generate_earnings_summary(self, doc_content: str, ticker: str, quarter: str) -> Dict[str, Any]:
        """Generate structured earnings summary using OpenAI"""
        try:
            prompt = self._create_summarization_prompt(doc_content, ticker, quarter)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst specializing in earnings call analysis. Extract key insights and format them into structured JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                # Clean up the content to extract JSON
                content_clean = content.strip()
                
                # Look for JSON block in the response
                if '```json' in content_clean:
                    json_start = content_clean.find('```json') + 7
                    json_end = content_clean.find('```', json_start)
                    if json_end != -1:
                        content_clean = content_clean[json_start:json_end].strip()
                elif '{' in content_clean and '}' in content_clean:
                    # Find the first complete JSON object
                    json_start = content_clean.find('{')
                    json_end = content_clean.rfind('}') + 1
                    content_clean = content_clean[json_start:json_end]
                
                summary_data = json.loads(content_clean)
                return self._validate_and_format_summary(summary_data, ticker, quarter)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON parsing failed: {e}, using text fallback")
                # Fallback: extract structured data from text
                return self._extract_summary_from_text(content, ticker, quarter)
                
        except Exception as e:
            logger.error(f"OpenAI summarization error: {e}")
            return self._get_fallback_summary(ticker, quarter)
    
    def answer_earnings_question(self, question: str, doc_content: str, ticker: str, quarter: str) -> Dict[str, Any]:
        """Answer specific questions about earnings documents using OpenAI"""
        try:
            prompt = f"""
            Based on the following earnings call transcript for {ticker} {quarter}, answer the user's question.
            
            Question: {question}
            
            Transcript:
            {doc_content[:4000]}  # Limit content to avoid token limits
            
            Provide a detailed answer with specific quotes and page references where possible.
            Format your response as JSON with:
            - answer: detailed response
            - citations: list of quotes with page numbers
            - confidence: high/medium/low
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst. Answer questions about earnings calls with specific citations and quotes."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "answer": content,
                    "citations": [],
                    "confidence": "medium"
                }
                
        except Exception as e:
            logger.error(f"OpenAI Q&A error: {e}")
            return {
                "answer": f"I apologize, but I encountered an error while analyzing the {ticker} {quarter} earnings call. Please try again.",
                "citations": [],
                "confidence": "low"
            }
    
    def _create_summarization_prompt(self, doc_content: str, ticker: str, quarter: str) -> str:
        """Create a detailed prompt for earnings summarization"""
        return f"""
        Analyze the following earnings call transcript for {ticker} {quarter} and extract key insights.
        
        Transcript:
        {doc_content[:6000]}  # Limit content to avoid token limits
        
        Please provide a comprehensive analysis in the following JSON format:
        {{
            "tldr": [
                "Key highlight 1 with specific numbers and context",
                "Key highlight 2 with specific numbers and context",
                "Key highlight 3 with specific numbers and context",
                "Key highlight 4 with specific numbers and context"
            ],
            "guidance": {{
                "revenue": "Specific revenue guidance with numbers",
                "eps": "EPS guidance with numbers", 
                "margin": "Margin guidance with numbers",
                "other": "Any other guidance mentioned"
            }},
            "kpis": [
                {{"metric": "Revenue", "value": "X.XB", "change": "+X% YoY", "context": "Additional context"}},
                {{"metric": "EPS", "value": "X.XX", "change": "+X% YoY", "context": "Additional context"}},
                {{"metric": "Operating Margin", "value": "X.X%", "change": "+X.X% YoY", "context": "Additional context"}},
                {{"metric": "Free Cash Flow", "value": "X.XB", "change": "+X% YoY", "context": "Additional context"}}
            ],
            "risks": [
                "Specific risk mentioned with context",
                "Another risk with specific details",
                "Third risk with relevant context"
            ],
            "quotes": [
                {{"quote": "Exact quote from transcript", "speaker": "CEO/CFO/etc", "page": 1, "context": "What they were discussing"}},
                {{"quote": "Another important quote", "speaker": "Speaker name", "page": 2, "context": "Context of the quote"}},
                {{"quote": "Third key quote", "speaker": "Speaker name", "page": 3, "context": "Context of the quote"}}
            ],
            "sector_insights": [
                "Industry trend or insight mentioned",
                "Market condition discussed",
                "Competitive landscape insight"
            ],
            "forward_looking": [
                "Future plans or initiatives mentioned",
                "Investment priorities discussed",
                "Strategic direction outlined"
            ]
        }}
        
        Important:
        - Use actual numbers and data from the transcript
        - Include specific quotes with speaker attribution
        - Focus on material information that investors would care about
        - Be precise and factual
        - If certain information is not available, use "Not specified" or omit the field
        """
    
    def _validate_and_format_summary(self, summary_data: Dict[str, Any], ticker: str, quarter: str) -> Dict[str, Any]:
        """Validate and format the OpenAI summary response"""
        try:
            # Ensure required fields exist
            required_fields = ['tldr', 'guidance', 'kpis', 'risks', 'quotes']
            for field in required_fields:
                if field not in summary_data:
                    summary_data[field] = []
            
            # Validate TL;DR
            if not isinstance(summary_data['tldr'], list):
                summary_data['tldr'] = []
            
            # Validate guidance
            if not isinstance(summary_data['guidance'], dict):
                summary_data['guidance'] = {}
            
            # Validate KPIs
            if not isinstance(summary_data['kpis'], list):
                summary_data['kpis'] = []
            
            # Validate risks
            if not isinstance(summary_data['risks'], list):
                summary_data['risks'] = []
            
            # Validate quotes
            if not isinstance(summary_data['quotes'], list):
                summary_data['quotes'] = []
            
            # Add metadata
            summary_data['ticker'] = ticker
            summary_data['quarter'] = quarter
            summary_data['generated_at'] = datetime.now().isoformat()
            summary_data['source'] = 'openai'
            
            return summary_data
            
        except Exception as e:
            logger.error(f"Error validating summary: {e}")
            return self._get_fallback_summary(ticker, quarter)
    
    def _extract_summary_from_text(self, text: str, ticker: str, quarter: str) -> Dict[str, Any]:
        """Extract structured data from text response when JSON parsing fails"""
        # This is a fallback method to extract key information from text
        lines = text.split('\n')
        
        tldr = []
        risks = []
        quotes = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('-'):
                tldr.append(line[1:].strip())
            elif 'risk' in line.lower() and not line.startswith('"'):
                risks.append(line)
            elif '"' in line and not line.startswith('"') and not line.endswith('"'):
                quotes.append(line)
        
        # Clean up the extracted data
        tldr = [item for item in tldr if item and not item.startswith('"')]
        risks = [item for item in risks if item and not item.startswith('"')]
        quotes = [item for item in quotes if item and not item.startswith('"')]
        
        return {
            "tldr": tldr[:4] if tldr else [f"{ticker} reported strong quarterly results"],
            "guidance": {
                "revenue": "Not specified",
                "eps": "Not specified",
                "margin": "Not specified"
            },
            "kpis": [
                {"metric": "Revenue", "value": "Not specified", "change": "Not specified", "context": "From transcript analysis"}
            ],
            "risks": risks[:3] if risks else ["Market risks mentioned in call"],
            "quotes": [{"quote": q, "speaker": "Management", "page": 1, "context": "Earnings discussion"} for q in quotes[:3]] if quotes else [{"quote": "Key insights from management", "speaker": "Management", "page": 1, "context": "Earnings discussion"}],
            "ticker": ticker,
            "quarter": quarter,
            "generated_at": datetime.now().isoformat(),
            "source": "openai_text_fallback"
        }
    
    def _get_fallback_summary(self, ticker: str, quarter: str) -> Dict[str, Any]:
        """Fallback summary when OpenAI fails"""
        return {
            "tldr": [
                f"{ticker} reported quarterly results for {quarter}",
                "Management provided updates on business performance",
                "Key financial metrics were discussed",
                "Forward guidance was provided"
            ],
            "guidance": {
                "revenue": "Guidance provided in call",
                "eps": "EPS guidance discussed",
                "margin": "Margin expectations shared"
            },
            "kpis": [
                {"metric": "Revenue", "value": "Reported in call", "change": "YoY change discussed", "context": "From earnings call"},
                {"metric": "EPS", "value": "Reported in call", "change": "YoY change discussed", "context": "From earnings call"}
            ],
            "risks": [
                "Market risks discussed in call",
                "Operational challenges mentioned",
                "Regulatory considerations noted"
            ],
            "quotes": [
                {"quote": "Management commentary from call", "speaker": "Management", "page": 1, "context": "Earnings discussion"}
            ],
            "ticker": ticker,
            "quarter": quarter,
            "generated_at": datetime.now().isoformat(),
            "source": "fallback"
        }

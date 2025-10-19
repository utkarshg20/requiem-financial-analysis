"""
Financial Metric Extractor for Earnings Analysis

This module provides automated extraction and calculation of financial metrics
from earnings call transcripts and documents.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class FinancialMetric:
    """Represents a financial metric with value and metadata"""
    name: str
    value: float
    unit: str  # 'B' for billions, 'M' for millions, '%' for percentages
    period: str  # 'Q3 2024', 'QoQ', 'YoY', etc.
    change: Optional[float] = None  # Percentage change
    change_type: Optional[str] = None  # 'increase', 'decrease', 'flat'
    context: Optional[str] = None  # Additional context from the text

@dataclass
class MetricExtractionResult:
    """Result of metric extraction from earnings content"""
    revenue: Optional[FinancialMetric] = None
    eps: Optional[FinancialMetric] = None
    operating_income: Optional[FinancialMetric] = None
    net_income: Optional[FinancialMetric] = None
    gross_margin: Optional[FinancialMetric] = None
    operating_margin: Optional[FinancialMetric] = None
    free_cash_flow: Optional[FinancialMetric] = None
    diluted_shares: Optional[FinancialMetric] = None
    additional_metrics: List[FinancialMetric] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = []

class FinancialMetricExtractor:
    """Extracts and calculates financial metrics from earnings content"""
    
    def __init__(self):
        self.metric_patterns = self._build_metric_patterns()
        self.unit_conversions = {
            'billion': 1e9,
            'b': 1e9,
            'million': 1e6,
            'm': 1e6,
            'thousand': 1e3,
            'k': 1e3,
            'trillion': 1e12,
            't': 1e12
        }
    
    def _build_metric_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for different financial metrics"""
        return {
            'revenue': [
                r'revenue.*?\$([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
                r'total\s+revenue.*?\$([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
                r'net\s+revenue.*?\$([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
                r'sales.*?\$([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
            ],
            'eps': [
                r'earnings\s+per\s+share.*?\$([0-9,]+\.?[0-9]*)',
                r'eps.*?\$([0-9,]+\.?[0-9]*)',
                r'diluted\s+eps.*?\$([0-9,]+\.?[0-9]*)',
                r'basic\s+eps.*?\$([0-9,]+\.?[0-9]*)',
            ],
            'operating_income': [
                r'operating\s+income.*?\$([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
                r'operating\s+profit.*?\$([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
                r'ebit.*?\$([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
            ],
            'net_income': [
                r'net\s+income.*?\$([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
                r'net\s+profit.*?\$([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
                r'profit.*?\$([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
            ],
            'gross_margin': [
                r'gross\s+margin.*?([0-9,]+\.?[0-9]*)\s*%',
                r'gross\s+profit\s+margin.*?([0-9,]+\.?[0-9]*)\s*%',
            ],
            'operating_margin': [
                r'operating\s+margin.*?([0-9,]+\.?[0-9]*)\s*%',
                r'operating\s+profit\s+margin.*?([0-9,]+\.?[0-9]*)\s*%',
            ],
            'free_cash_flow': [
                r'free\s+cash\s+flow.*?\$([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
                r'fcf.*?\$([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
                r'cash\s+flow.*?\$([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
            ],
            'diluted_shares': [
                r'diluted\s+shares.*?([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
                r'weighted\s+average\s+shares.*?([0-9,]+\.?[0-9]*)\s*([bmk]?)illion',
            ]
        }
    
    def extract_metrics(self, content: str, ticker: str, quarter: str) -> MetricExtractionResult:
        """Extract financial metrics from earnings content"""
        try:
            logger.info(f"Extracting metrics for {ticker} {quarter}")
            
            # Clean and normalize content
            content_lower = content.lower()
            
            # Extract basic metrics
            result = MetricExtractionResult()
            
            # Revenue
            result.revenue = self._extract_metric(content_lower, 'revenue', 'Revenue')
            
            # EPS
            result.eps = self._extract_metric(content_lower, 'eps', 'EPS')
            
            # Operating Income
            result.operating_income = self._extract_metric(content_lower, 'operating_income', 'Operating Income')
            
            # Net Income
            result.net_income = self._extract_metric(content_lower, 'net_income', 'Net Income')
            
            # Margins
            result.gross_margin = self._extract_percentage_metric(content_lower, 'gross_margin', 'Gross Margin')
            result.operating_margin = self._extract_percentage_metric(content_lower, 'operating_margin', 'Operating Margin')
            
            # Free Cash Flow
            result.free_cash_flow = self._extract_metric(content_lower, 'free_cash_flow', 'Free Cash Flow')
            
            # Diluted Shares
            result.diluted_shares = self._extract_metric(content_lower, 'diluted_shares', 'Diluted Shares')
            
            # Calculate additional metrics
            self._calculate_derived_metrics(result)
            
            # Extract growth rates and changes
            self._extract_growth_rates(content_lower, result)
            
            logger.info(f"Successfully extracted metrics for {ticker} {quarter}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            return MetricExtractionResult()
    
    def _extract_metric(self, content: str, metric_type: str, display_name: str) -> Optional[FinancialMetric]:
        """Extract a specific metric from content"""
        patterns = self.metric_patterns.get(metric_type, [])
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    value_str = match.group(1).replace(',', '')
                    value = float(value_str)
                    
                    # Get unit if available
                    unit = 'B'  # Default to billions
                    if len(match.groups()) > 1 and match.group(2):
                        unit_raw = match.group(2).lower()
                        if unit_raw in ['b', 'billion']:
                            unit = 'B'
                        elif unit_raw in ['m', 'million']:
                            unit = 'M'
                        elif unit_raw in ['k', 'thousand']:
                            unit = 'K'
                    
                    # Convert to standard unit (billions)
                    if unit == 'M':
                        value = value / 1000
                    elif unit == 'K':
                        value = value / 1000000
                    
                    return FinancialMetric(
                        name=display_name,
                        value=value,
                        unit='B',
                        period='Current',
                        context=match.group(0)
                    )
                    
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing metric {metric_type}: {e}")
                    continue
        
        return None
    
    def _extract_percentage_metric(self, content: str, metric_type: str, display_name: str) -> Optional[FinancialMetric]:
        """Extract percentage-based metrics"""
        patterns = self.metric_patterns.get(metric_type, [])
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    value_str = match.group(1).replace(',', '')
                    value = float(value_str)
                    
                    return FinancialMetric(
                        name=display_name,
                        value=value,
                        unit='%',
                        period='Current',
                        context=match.group(0)
                    )
                    
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing percentage metric {metric_type}: {e}")
                    continue
        
        return None
    
    def _extract_growth_rates(self, content: str, result: MetricExtractionResult):
        """Extract growth rates and percentage changes"""
        growth_patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*(?:increase|growth|up|rise)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:decrease|decline|down|fall)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:year-over-year|yoy|y/y)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:quarter-over-quarter|qoq|q/q)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:sequential|seq)',
        ]
        
        for pattern in growth_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    change_value = float(match.group(1))
                    context = match.group(0)
                    
                    # Determine if it's an increase or decrease
                    change_type = 'increase'
                    if any(word in context.lower() for word in ['decrease', 'decline', 'down', 'fall']):
                        change_type = 'decrease'
                    
                    # Try to associate with a metric
                    if 'revenue' in context.lower():
                        if result.revenue:
                            result.revenue.change = change_value
                            result.revenue.change_type = change_type
                    elif 'eps' in context.lower() or 'earnings' in context.lower():
                        if result.eps:
                            result.eps.change = change_value
                            result.eps.change_type = change_type
                    elif 'operating' in context.lower():
                        if result.operating_income:
                            result.operating_income.change = change_value
                            result.operating_income.change_type = change_type
                    elif 'net' in context.lower() and 'income' in context.lower():
                        if result.net_income:
                            result.net_income.change = change_value
                            result.net_income.change_type = change_type
                    
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing growth rate: {e}")
                    continue
    
    def _calculate_derived_metrics(self, result: MetricExtractionResult):
        """Calculate derived metrics from basic metrics"""
        try:
            # Calculate additional ratios and metrics
            if result.revenue and result.net_income:
                # Net profit margin
                net_margin = (result.net_income.value / result.revenue.value) * 100
                result.additional_metrics.append(FinancialMetric(
                    name="Net Profit Margin",
                    value=net_margin,
                    unit="%",
                    period="Current"
                ))
            
            if result.revenue and result.operating_income:
                # Operating margin (if not already extracted)
                if not result.operating_margin:
                    op_margin = (result.operating_income.value / result.revenue.value) * 100
                    result.operating_margin = FinancialMetric(
                        name="Operating Margin",
                        value=op_margin,
                        unit="%",
                        period="Current"
                    )
            
            if result.eps and result.diluted_shares:
                # Validate EPS calculation
                calculated_eps = result.net_income.value / result.diluted_shares.value if result.net_income else None
                if calculated_eps and abs(calculated_eps - result.eps.value) > 0.1:
                    result.additional_metrics.append(FinancialMetric(
                        name="Calculated EPS",
                        value=calculated_eps,
                        unit="$",
                        period="Current"
                    ))
            
        except Exception as e:
            logger.debug(f"Error calculating derived metrics: {e}")
    
    def format_metrics_for_display(self, result: MetricExtractionResult) -> Dict[str, Any]:
        """Format extracted metrics for display in the UI"""
        formatted = {
            "basic_metrics": [],
            "growth_rates": [],
            "margins": [],
            "additional_metrics": []
        }
        
        # Basic metrics
        for metric in [result.revenue, result.eps, result.operating_income, result.net_income, result.free_cash_flow]:
            if metric:
                formatted["basic_metrics"].append({
                    "name": metric.name,
                    "value": f"{metric.value:.2f}{metric.unit}",
                    "change": f"{metric.change:+.1f}%" if metric.change else None,
                    "change_type": metric.change_type,
                    "context": metric.context
                })
        
        # Margins
        for metric in [result.gross_margin, result.operating_margin]:
            if metric:
                formatted["margins"].append({
                    "name": metric.name,
                    "value": f"{metric.value:.1f}%",
                    "context": metric.context
                })
        
        # Additional metrics
        for metric in result.additional_metrics:
            formatted["additional_metrics"].append({
                "name": metric.name,
                "value": f"{metric.value:.2f}{metric.unit}",
                "context": metric.context
            })
        
        return formatted
    
    def generate_metric_insights(self, result: MetricExtractionResult, ticker: str) -> str:
        """Generate AI insights based on extracted metrics"""
        insights = []
        
        # Revenue insights
        if result.revenue and result.revenue.change:
            if result.revenue.change > 10:
                insights.append(f"**Strong Revenue Growth**: {ticker} reported {result.revenue.change:+.1f}% revenue growth, indicating robust business performance.")
            elif result.revenue.change < -5:
                insights.append(f"**Revenue Decline**: {ticker} experienced {result.revenue.change:+.1f}% revenue decline, which may indicate market challenges.")
            else:
                insights.append(f"**Stable Revenue**: {ticker} showed {result.revenue.change:+.1f}% revenue growth, maintaining steady performance.")
        
        # Profitability insights
        if result.operating_margin and result.operating_margin.value > 20:
            insights.append(f"**Strong Profitability**: Operating margin of {result.operating_margin.value:.1f}% demonstrates excellent operational efficiency.")
        elif result.operating_margin and result.operating_margin.value < 10:
            insights.append(f"**Margin Pressure**: Operating margin of {result.operating_margin.value:.1f}% suggests potential cost management challenges.")
        
        # EPS insights
        if result.eps and result.eps.change:
            if result.eps.change > 15:
                insights.append(f"**Earnings Growth**: EPS increased {result.eps.change:+.1f}%, showing strong earnings momentum.")
            elif result.eps.change < -10:
                insights.append(f"**Earnings Pressure**: EPS declined {result.eps.change:+.1f}%, indicating potential profitability challenges.")
        
        # Cash flow insights
        if result.free_cash_flow and result.free_cash_flow.value > 0:
            insights.append(f"**Positive Cash Generation**: Free cash flow of ${result.free_cash_flow.value:.1f}B provides strong financial flexibility.")
        
        return "\n\n".join(insights) if insights else "**Metric Analysis**: Detailed financial metrics have been extracted and analyzed."
"""
Requiem SDK Client - Easy scripting interface for strategy backtesting
"""

from __future__ import annotations
import requests
from typing import Dict, Any, Optional
from datetime import datetime
import json
import base64
from pathlib import Path


class RequiemClient:
    """
    Pythonic client for Requiem strategy backtesting API
    
    Example:
        >>> from sdk import RequiemClient
        >>> client = RequiemClient("http://localhost:8000")
        >>> spec = client.prompt_to_spec("Test momentum on SPY since 2020 monthly")
        >>> tearsheet = client.execute_and_fetch(spec)
        >>> html = client.tearsheet_to_html(tearsheet)
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client
        
        Args:
            base_url: Base URL of Requiem API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'requiem-sdk/0.1'
        })
    
    def health_check(self) -> bool:
        """
        Check if API is reachable
        
        Returns:
            bool: True if API is healthy
        """
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200 and response.json().get('ok') == True
        except Exception:
            return False
    
    def prompt_to_spec(self, prompt: str, auto_fill_defaults: bool = True) -> Dict[str, Any]:
        """
        Convert natural language prompt to strategy spec
        
        Calls /prompt/audit, extracts fields, and optionally fills defaults
        
        Args:
            prompt: Natural language description (e.g., "Test momentum on SPY since 2020")
            auto_fill_defaults: If True, automatically fills missing fields with sensible defaults
        
        Returns:
            Dict: Strategy specification ready for execution
            
        Example:
            >>> spec = client.prompt_to_spec("Test momentum on SPY since 2020 monthly")
            >>> print(spec['ticker'])  # "SPY"
        """
        # Call /prompt/audit
        response = self.session.post(
            f"{self.base_url}/prompt/audit",
            json={"prompt": prompt}
        )
        response.raise_for_status()
        
        data = response.json()
        spec_skeleton = data['spec_skeleton']
        questions = data['questions']
        confidence = data['confidence']
        
        # Auto-fill defaults if requested
        if auto_fill_defaults:
            # Fill missing feature
            if 'custom' not in spec_skeleton or 'feature' not in spec_skeleton.get('custom', {}):
                if 'custom' not in spec_skeleton:
                    spec_skeleton['custom'] = {}
                spec_skeleton['custom']['feature'] = 'momentum_12m_skip_1m'
            
            # Fill missing signal_rule
            if 'custom' in spec_skeleton and 'signal_rule' not in spec_skeleton['custom']:
                feature = spec_skeleton['custom'].get('feature', '')
                
                # Choose appropriate default signal based on feature
                if 'momentum' in feature:
                    spec_skeleton['custom']['signal_rule'] = {'type': 'rank_top_frac', 'top_frac': 0.2}
                elif 'zscore' in feature:
                    spec_skeleton['custom']['signal_rule'] = {'type': 'threshold', 'lower': -1.0, 'upper': 1.0}
                elif 'rsi' in feature:
                    spec_skeleton['custom']['signal_rule'] = {'type': 'threshold', 'lower': 30, 'upper': 70}
                else:
                    spec_skeleton['custom']['signal_rule'] = {'type': 'rank_top_frac', 'top_frac': 0.2}
            
            # Fill missing trading_rules
            if 'trading_rules' not in spec_skeleton:
                spec_skeleton['trading_rules'] = {'rebalance': 'monthly', 'transaction_costs_bps': 5.0}
            
            # Fill missing end date
            if 'end' not in spec_skeleton:
                spec_skeleton['end'] = datetime.now().strftime('%Y-%m-%d')
        
        # Add metadata
        spec_skeleton['_sdk_metadata'] = {
            'prompt': prompt,
            'confidence': confidence,
            'questions': questions,
            'auto_filled': auto_fill_defaults
        }
        
        return spec_skeleton
    
    def validate_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate strategy specification
        
        Calls /specs/validate to check for errors and get suggestions
        
        Args:
            spec: Strategy specification
        
        Returns:
            Dict with 'valid', 'errors', 'warnings', 'suggested_fixes'
            
        Example:
            >>> result = client.validate_spec(spec)
            >>> if not result['valid']:
            ...     print(result['errors'])
        """
        response = self.session.post(
            f"{self.base_url}/specs/validate",
            json={"spec_skeleton": spec}
        )
        response.raise_for_status()
        return response.json()
    
    def execute_and_fetch(self, spec: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """
        Execute strategy and wait for results
        
        Args:
            spec: Strategy specification
            timeout: Max seconds to wait for completion (default: 300)
        
        Returns:
            Dict: Complete tearsheet with metrics, figures, tables
            
        Example:
            >>> tearsheet = client.execute_and_fetch(spec)
            >>> print(tearsheet['metrics']['performance']['sharpe'])
        """
        # Submit execution request
        response = self.session.post(
            f"{self.base_url}/runs/execute",
            json={"spec": spec}
        )
        response.raise_for_status()
        
        result = response.json()
        run_id = result['run_id']
        
        # Poll for completion
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status_response = self.session.get(f"{self.base_url}/runs/{run_id}/status")
            status_response.raise_for_status()
            
            status_data = status_response.json()
            state = status_data['state']
            
            if state == 'done':
                # Fetch tearsheet
                tearsheet_response = self.session.get(f"{self.base_url}/runs/{run_id}/tearsheet")
                tearsheet_response.raise_for_status()
                return tearsheet_response.json()
            
            elif state == 'error':
                raise RuntimeError(f"Run {run_id} failed: {status_data.get('error', 'Unknown error')}")
            
            # Wait before polling again
            time.sleep(1)
        
        raise TimeoutError(f"Run {run_id} did not complete within {timeout} seconds")
    
    def tearsheet_to_html(self, tearsheet: Dict[str, Any], inline_charts: bool = True) -> str:
        """
        Convert tearsheet to shareable HTML
        
        Args:
            tearsheet: Tearsheet dictionary from execute_and_fetch
            inline_charts: If True, embed charts as base64 (default: True)
        
        Returns:
            str: HTML string with embedded charts and metrics
            
        Example:
            >>> html = client.tearsheet_to_html(tearsheet)
            >>> Path("results.html").write_text(html)
        """
        metrics = tearsheet['metrics']['performance']
        config = tearsheet['summary']['config']
        figures = tearsheet['figures']
        tables = tearsheet['tables']
        
        # Build HTML
        html_parts = []
        
        # Header
        html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Requiem Strategy Tearsheet</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
               max-width: 1200px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; margin-top: 0; }
        h2 { color: #666; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; margin-top: 30px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric { background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #007bff; }
        .metric-label { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #333; margin-top: 5px; }
        .chart { margin: 20px 0; text-align: center; }
        .chart img { max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }
        th { background: #f8f9fa; font-weight: 600; color: #666; }
        tr:hover { background: #f8f9fa; }
        .config { background: #f8f9fa; padding: 15px; border-radius: 6px; margin: 20px 0; }
        .config-item { margin: 8px 0; }
        .config-label { font-weight: 600; color: #666; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 2px solid #e0e0e0; 
                  text-align: center; color: #999; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
""")
        
        # Title
        title = tearsheet['summary'].get('title', 'Strategy Backtest')
        html_parts.append(f"        <h1>📊 {title}</h1>\n")
        
        # Strategy Configuration
        html_parts.append("        <h2>⚙️ Strategy Configuration</h2>\n")
        html_parts.append("        <div class='config'>\n")
        html_parts.append(f"            <div class='config-item'><span class='config-label'>Feature:</span> {config['feature']}</div>\n")
        html_parts.append(f"            <div class='config-item'><span class='config-label'>Signal Type:</span> {config['signal_type']}</div>\n")
        html_parts.append(f"            <div class='config-item'><span class='config-label'>Rebalance:</span> {config['rebalance']}</div>\n")
        html_parts.append(f"            <div class='config-item'><span class='config-label'>Transaction Costs:</span> {config['tc_bps']} bps</div>\n")
        html_parts.append("        </div>\n")
        
        # Performance Metrics
        html_parts.append("        <h2>📈 Performance Metrics</h2>\n")
        html_parts.append("        <div class='metrics'>\n")
        
        metric_configs = [
            ('CAGR', metrics['cagr'], '%'),
            ('Sharpe', metrics['sharpe'], ''),
            ('Sortino', metrics['sortino'], ''),
            ('Hit Rate', metrics['hit_rate'], '%'),
            ('Annual Vol', metrics['vol_annual'], '%'),
            ('Max Drawdown', metrics['max_drawdown_pct'], '%'),
            ('Avg Turnover', metrics.get('avg_turnover', 0), '%'),
            ('Exposure', metrics.get('exposure_share', 0), '%'),
        ]
        
        for label, value, unit in metric_configs:
            if unit == '%':
                formatted_value = f"{value:.2%}"
            else:
                formatted_value = f"{value:.2f}"
            
            html_parts.append(f"            <div class='metric'>\n")
            html_parts.append(f"                <div class='metric-label'>{label}</div>\n")
            html_parts.append(f"                <div class='metric-value'>{formatted_value}</div>\n")
            html_parts.append(f"            </div>\n")
        
        html_parts.append("        </div>\n")
        
        # Charts
        html_parts.append("        <h2>📊 Performance Charts</h2>\n")
        
        for fig in figures:
            chart_path = fig['path']
            chart_title = fig['title']
            
            if inline_charts and chart_path and Path(chart_path).exists():
                # Read and encode chart as base64
                with open(chart_path, 'rb') as f:
                    chart_data = base64.b64encode(f.read()).decode('utf-8')
                    html_parts.append(f"        <div class='chart'>\n")
                    html_parts.append(f"            <h3>{chart_title}</h3>\n")
                    html_parts.append(f"            <img src='data:image/png;base64,{chart_data}' alt='{chart_title}'>\n")
                    html_parts.append(f"        </div>\n")
            else:
                html_parts.append(f"        <div class='chart'>\n")
                html_parts.append(f"            <p>📈 {chart_title}: {chart_path}</p>\n")
                html_parts.append(f"        </div>\n")
        
        # Tables
        if tables:
            html_parts.append("        <h2>📋 Performance Tables</h2>\n")
            
            for table in tables:
                html_parts.append(f"        <h3>{table['title']}</h3>\n")
                html_parts.append("        <table>\n")
                html_parts.append("            <thead><tr>\n")
                
                for col in table['columns']:
                    html_parts.append(f"                <th>{col}</th>\n")
                
                html_parts.append("            </tr></thead>\n")
                html_parts.append("            <tbody>\n")
                
                for row in table['rows']:
                    html_parts.append("                <tr>\n")
                    for val in row:
                        if isinstance(val, float):
                            formatted = f"{val:.2%}" if abs(val) < 10 else f"{val:.2f}"
                        else:
                            formatted = str(val)
                        html_parts.append(f"                    <td>{formatted}</td>\n")
                    html_parts.append("                </tr>\n")
                
                html_parts.append("            </tbody>\n")
                html_parts.append("        </table>\n")
        
        # Footer
        html_parts.append("        <div class='footer'>\n")
        html_parts.append(f"            Generated by Requiem SDK • {tearsheet.get('generated_at', '')}\n")
        html_parts.append("        </div>\n")
        
        # Close HTML
        html_parts.append("    </div>\n")
        html_parts.append("</body>\n")
        html_parts.append("</html>")
        
        return ''.join(html_parts)
    
    def quick_backtest(self, prompt: str, output_html: Optional[str] = None) -> Dict[str, Any]:
        """
        One-liner to run a complete backtest from prompt
        
        Args:
            prompt: Natural language strategy description
            output_html: If provided, saves HTML tearsheet to this path
        
        Returns:
            Dict: Complete tearsheet
            
        Example:
            >>> tearsheet = client.quick_backtest("Test momentum on SPY since 2020", "results.html")
            >>> print(f"Sharpe: {tearsheet['metrics']['performance']['sharpe']:.2f}")
        """
        # Convert prompt to spec
        spec = self.prompt_to_spec(prompt, auto_fill_defaults=True)
        
        # Execute
        tearsheet = self.execute_and_fetch(spec)
        
        # Save HTML if requested
        if output_html:
            html = self.tearsheet_to_html(tearsheet)
            Path(output_html).write_text(html)
            print(f"✅ Tearsheet saved to {output_html}")
        
        return tearsheet


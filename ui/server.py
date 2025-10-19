#!/usr/bin/env python3
"""
Simple HTTP server to serve the Requiem UI
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

def serve_ui(port=3000):
    """Serve the UI on the specified port"""
    
    # Change to the UI directory
    ui_dir = Path(__file__).parent
    os.chdir(ui_dir)
    
    # Create server
    Handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"🚀 Serving Requiem UI at http://localhost:{port}")
        print(f"📁 Serving files from: {ui_dir}")
        print(f"🔗 Make sure Requiem API is running at http://localhost:8000")
        print(f"\nPress Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n✅ Server stopped")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    serve_ui(port)

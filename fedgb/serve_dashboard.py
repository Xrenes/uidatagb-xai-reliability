"""
serve_dashboard.py
------------------
Starts a local HTTP server so the live dashboard works correctly.

Run from the fedgb/ folder:
    python serve_dashboard.py

Then open:  http://localhost:8765
"""
import http.server
import os
import socketserver
import webbrowser

PORT    = 8765
SERVE_DIR = os.path.join(os.path.dirname(__file__), "outputs")

os.makedirs(SERVE_DIR, exist_ok=True)
os.chdir(SERVE_DIR)

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

print(f"\n  FedGB Live Dashboard")
print(f"  ─────────────────────────────────")
print(f"  Serving:  {SERVE_DIR}")
print(f"  Open:     http://localhost:{PORT}/dashboard.html")
print(f"  Press Ctrl+C to stop.\n")

webbrowser.open(f"http://localhost:{PORT}/dashboard.html")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")

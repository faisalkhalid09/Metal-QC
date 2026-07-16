import os
import sys
import threading
import webbrowser
from werkzeug.serving import make_server
from app import app

def run_server(host='0.0.0.0', start_port=5731):
    port = start_port
    server = None
    while port < start_port + 10:
        try:
            server = make_server(host, port, app)
            break
        except OSError as e:
            if "Address already in use" in str(e) or getattr(e, 'errno', None) in (98, 10048):
                print(f"Port {port} is in use, trying {port+1}...")
                port += 1
            else:
                raise e
    
    if not server:
        print(f"Error: Could not bind to any port between {start_port} and {start_port+9}.")
        sys.exit(1)
        
    print(f"Server started. Access locally at http://localhost:{port}")
    if host == '0.0.0.0':
        print(f"Also accessible on your network via your machine's IP address.")
    
    # Auto-open browser
    def open_browser():
        webbrowser.open(f'http://localhost:{port}')
        
    # Open browser after a small delay
    timer = threading.Timer(1.0, open_browser)
    timer.start()
    
    server.serve_forever()

if __name__ == '__main__':
    run_server()

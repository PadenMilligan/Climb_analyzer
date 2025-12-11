"""
Helper script to start the Flask server and display connection info
for accessing from your iPhone on the same network.
"""
import socket
import subprocess
import sys
import os

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def main():
    print("=" * 60)
    print("🧗 Climb Analyzer - Server Starting")
    print("=" * 60)
    
    local_ip = get_local_ip()
    port = 5000
    
    print(f"\n📱 To access from your iPhone:")
    print(f"   1. Make sure your iPhone is on the SAME WiFi network")
    print(f"   2. Open Safari on your iPhone")
    print(f"   3. Go to: http://{local_ip}:{port}")
    print(f"\n💡 Alternative: http://{local_ip}:{port}/")
    print(f"\n⚠️  Note: Keep this window open while using the app")
    print("=" * 60)
    print("\nStarting server...\n")
    
    # Change to the directory containing app.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Import and run the app
    try:
        from app import app
        app.run(debug=False, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Dashboard Launcher for GraphRAG Risk Analysis
"""

import subprocess
import sys
from pathlib import Path

def install_streamlit():
    """Install streamlit if not available"""
    try:
        import streamlit
        return True
    except ImportError:
        print("📦 Installing Streamlit...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
        return True

def main():
    """Launch the Streamlit dashboard"""
    
    print("🚀 Launching GraphRAG Risk Analysis Dashboard...")
    
    # Ensure streamlit is installed
    if not install_streamlit():
        print("❌ Failed to install Streamlit")
        return
    
    # Check if dashboard script exists
    dashboard_path = Path("scripts/interactive_dashboard.py")
    if not dashboard_path.exists():
        print(f"❌ Dashboard script not found: {dashboard_path}")
        return
    
    # Launch dashboard
    try:
        print("🌐 Starting dashboard server...")
        print("📊 Dashboard will open in your browser at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the dashboard")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    main()
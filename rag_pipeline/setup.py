#!/usr/bin/env python3
"""
Setup Script for RAG Pipeline
=============================

This script sets up the RAG pipeline environment and dependencies.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("📦 Installing RAG Pipeline dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating necessary directories...")
    directories = ["./chroma_db", "./logs"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def check_openai_key():
    """Check if OpenAI API key is configured."""
    print("🔑 Checking OpenAI API key configuration...")
    
    # Check environment variable
    if os.getenv('OPENAI_API_KEY'):
        print("   ✅ OpenAI API key found in environment variables")
        return True
    
    # Check config file
    if os.path.exists('config.yml'):
        import yaml
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
            if config.get('openai_api_key') and config['openai_api_key'] != "your-openai-api-key-here":
                print("   ✅ OpenAI API key found in config.yml")
                return True
    
    print("   ⚠️  OpenAI API key not found")
    print("   📝 Please set your API key using one of these methods:")
    print("      - Set OPENAI_API_KEY environment variable")
    print("      - Edit config.yml and add your API key")
    return False

def main():
    """Main setup function."""
    print("🚀 RAG Pipeline Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during dependency installation")
        return
    
    # Create directories
    create_directories()
    
    # Check OpenAI key
    check_openai_key()
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Set your OpenAI API key (if not already done)")
    print("2. Run: python cli.py --ingest 'path/to/your/csv/file'")
    print("3. Run: python cli.py --chat")
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main()

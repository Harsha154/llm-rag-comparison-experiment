#!/usr/bin/env python3
"""
RAG Pipeline Command Line Interface
==================================

This module provides a command-line interface for the RAG pipeline,
allowing users to ingest CSV data and chat with the vector database.
"""

import os
import sys
import argparse
from rag_pipeline import RAGPipeline

def interactive_chat(rag: RAGPipeline) -> None:
    """Start an interactive chat session with the RAG system."""
    print("\n" + "="*60)
    print("🤖 Insurance RAG Chat System")
    print("="*60)
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'help' for available commands")
    print("Type 'info' to see collection information")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\n📋 Available Commands:")
                print("- Ask questions about insurance scenarios")
                print("- Request advice for specific situations")
                print("- Compare different insurance options")
                print("- Type 'info' to see collection details")
                print("- Type 'quit' to exit")
                continue
            
            if user_input.lower() == 'info':
                info = rag.get_collection_info()
                print(f"\n📊 Collection Information:")
                print(f"   Name: {info.get('collection_name', 'N/A')}")
                print(f"   Documents: {info.get('document_count', 'N/A')}")
                print(f"   Directory: {info.get('persist_directory', 'N/A')}")
                continue
            
            if not user_input:
                continue
            
            print("\n🤖 AI: ", end="", flush=True)
            response = rag.chat_with_rag(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="RAG Pipeline for Insurance Dataset")
    parser.add_argument("--ingest", type=str, help="Path to CSV file to ingest")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat session")
    parser.add_argument("--query", type=str, help="Single query to run")
    parser.add_argument("--collection", type=str, default="insurance_scenarios", 
                       help="ChromaDB collection name")
    parser.add_argument("--db-path", type=str, default="./chroma_db", 
                       help="ChromaDB persistence directory")
    parser.add_argument("--delete", action="store_true", help="Delete the collection")
    parser.add_argument("--info", action="store_true", help="Show collection information")
    
    args = parser.parse_args()
    
    if not any([args.ingest, args.chat, args.query, args.delete, args.info]):
        parser.print_help()
        return
    
    # Initialize RAG pipeline
    try:
        rag = RAGPipeline(collection_name=args.collection, persist_directory=args.db_path)
    except Exception as e:
        print(f"❌ Error initializing RAG pipeline: {e}")
        return
    
    # Delete collection if requested
    if args.delete:
        try:
            rag.delete_collection()
            print("✅ Collection deleted successfully!")
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
        return
    
    # Show collection info if requested
    if args.info:
        try:
            info = rag.get_collection_info()
            print(f"\n📊 Collection Information:")
            print(f"   Name: {info.get('collection_name', 'N/A')}")
            print(f"   Documents: {info.get('document_count', 'N/A')}")
            print(f"   Directory: {info.get('persist_directory', 'N/A')}")
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
        return
    
    # Ingest CSV if specified
    if args.ingest:
        if not os.path.exists(args.ingest):
            print(f"❌ Error: CSV file not found: {args.ingest}")
            return
        
        print(f"📥 Ingesting CSV file: {args.ingest}")
        try:
            rag.ingest_csv(args.ingest)
            print("✅ CSV ingestion completed!")
        except Exception as e:
            print(f"❌ Error during ingestion: {e}")
            return
    
    # Run single query if specified
    if args.query:
        try:
            print(f"🔍 Query: {args.query}")
            response = rag.chat_with_rag(args.query)
            print(f"\n🤖 Response: {response}")
        except Exception as e:
            print(f"❌ Error running query: {e}")
        return
    
    # Start chat if specified
    if args.chat:
        interactive_chat(rag)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Optimized RAG Pipeline Command Line Interface
============================================

This module provides a command-line interface for the optimized RAG pipeline,
with batch processing and parallelization for handling large datasets efficiently.
"""

import os
import sys
import argparse
import time
from rag_pipeline_optimized import OptimizedRAGPipeline

def interactive_chat(rag: OptimizedRAGPipeline) -> None:
    """Start an interactive chat session with the RAG system."""
    print("\n" + "="*60)
    print("🤖 Optimized Insurance RAG Chat System")
    print("="*60)
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'help' for available commands")
    print("Type 'info' to see collection information")
    print("Type 'config' to see processing configuration")
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
                print("- Type 'config' to see processing settings")
                print("- Type 'quit' to exit")
                continue
            
            if user_input.lower() == 'info':
                info = rag.get_collection_info()
                print(f"\n📊 Collection Information:")
                print(f"   Name: {info.get('collection_name', 'N/A')}")
                print(f"   Documents: {info.get('document_count', 'N/A'):,}")
                print(f"   Directory: {info.get('persist_directory', 'N/A')}")
                continue
            
            if user_input.lower() == 'config':
                print(f"\n⚙️  Processing Configuration:")
                print(f"   Batch Size: {rag.batch_size:,} documents per batch")
                print(f"   Max Workers: {rag.max_workers} parallel workers")
                print(f"   Max Retries: {rag.max_retries} retries per request")
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
    parser = argparse.ArgumentParser(description="Optimized RAG Pipeline for Insurance Dataset")
    parser.add_argument("--ingest", type=str, help="Path to CSV file to ingest")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat session")
    parser.add_argument("--query", type=str, help="Single query to run")
    parser.add_argument("--collection", type=str, default="insurance_scenarios", 
                       help="ChromaDB collection name")
    parser.add_argument("--db-path", type=str, default="./chroma_db", 
                       help="ChromaDB persistence directory")
    parser.add_argument("--delete", action="store_true", help="Delete the collection")
    parser.add_argument("--info", action="store_true", help="Show collection information")
    parser.add_argument("--batch-size", type=int, default=1000, 
                       help="Number of documents per batch (default: 1000)")
    parser.add_argument("--workers", type=int, default=10, 
                       help="Number of parallel workers (default: 10)")
    parser.add_argument("--test-performance", action="store_true", 
                       help="Test performance with sample data")
    
    args = parser.parse_args()
    
    if not any([args.ingest, args.chat, args.query, args.delete, args.info, args.test_performance]):
        parser.print_help()
        return
    
    # Initialize optimized RAG pipeline
    try:
        rag = OptimizedRAGPipeline(
            collection_name=args.collection, 
            persist_directory=args.db_path
        )
        
        # Override configuration if provided
        if args.batch_size != 1000:
            rag.batch_size = args.batch_size
        if args.workers != 10:
            rag.max_workers = args.workers
            
        print(f"🚀 Optimized RAG Pipeline initialized")
        print(f"   Batch Size: {rag.batch_size:,} documents")
        print(f"   Workers: {rag.max_workers} parallel workers")
        print(f"   Collection: {rag.collection_name}")
        
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
            print(f"   Documents: {info.get('document_count', 'N/A'):,}")
            print(f"   Directory: {info.get('persist_directory', 'N/A')}")
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
        return
    
    # Test performance if requested
    if args.test_performance:
        try:
            print("🧪 Testing performance with sample data...")
            # Create sample data for testing
            sample_data = []
            for i in range(1000):
                sample_data.append([
                    f"INS_{i}",  # Probe ID
                    "25",        # Age
                    "Male",      # Gender
                    "Single",    # Marital Status
                    "0",         # Children
                    "50000",     # Income
                    "Good",      # Health Status
                    "Engineer",  # Occupation
                    "California", # Location
                    "1600",      # Option 1
                    "1200",      # Option 2
                    "800",       # Option 3
                    "0",         # Option 4
                    "3"          # Ground Truth
                ])
            
            # Save sample data to temporary CSV
            import pandas as pd
            import tempfile
            
            df = pd.DataFrame(sample_data, columns=[
                'Probe ID', 'Age', 'Gender', 'Marital Status', 'Children',
                'Income', 'Health Status', 'Occupation', 'Location',
                'Option 1', 'Option 2', 'Option 3', 'Option 4', 'Ground Truth'
            ])
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                temp_csv = f.name
            
            print(f"📊 Created test dataset with {len(sample_data):,} rows")
            
            # Test ingestion performance
            start_time = time.time()
            rag.ingest_csv_optimized(temp_csv)
            end_time = time.time()
            
            processing_time = end_time - start_time
            speed = len(sample_data) / processing_time
            
            print(f"\n📈 Performance Results:")
            print(f"   Documents processed: {len(sample_data):,}")
            print(f"   Processing time: {processing_time:.2f} seconds")
            print(f"   Speed: {speed:.2f} documents/second")
            print(f"   Estimated time for 69,000 rows: {69000/speed/60:.1f} minutes")
            
            # Clean up
            os.unlink(temp_csv)
            
        except Exception as e:
            print(f"❌ Error during performance test: {e}")
        return
    
    # Ingest CSV if specified
    if args.ingest:
        if not os.path.exists(args.ingest):
            print(f"❌ Error: CSV file not found: {args.ingest}")
            return
        
        print(f"📥 Ingesting CSV file: {args.ingest}")
        print(f"⚙️  Using batch size: {rag.batch_size:,} documents")
        print(f"⚙️  Using {rag.max_workers} parallel workers")
        
        try:
            start_time = time.time()
            rag.ingest_csv_optimized(args.ingest)
            end_time = time.time()
            
            processing_time = end_time - start_time
            print(f"\n✅ CSV ingestion completed!")
            print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
            
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

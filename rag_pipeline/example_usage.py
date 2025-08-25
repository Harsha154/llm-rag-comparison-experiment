#!/usr/bin/env python3
"""
RAG Pipeline Example Usage
==========================

This script demonstrates how to use the RAG pipeline programmatically.
"""

import os
from rag_pipeline import RAGPipeline

def example_usage():
    """Example usage of the RAG pipeline."""
    
    # Initialize the RAG pipeline
    print("🚀 Initializing RAG Pipeline...")
    rag = RAGPipeline(
        collection_name="example_insurance",
        persist_directory="./example_chroma_db"
    )
    
    # Example 1: Ingest CSV data
    csv_path = "../data/dataset2_to be used in experiments with target --includes ground-truth.csv"
    if os.path.exists(csv_path):
        print(f"\n📥 Example 1: Ingesting CSV data from {csv_path}")
        try:
            rag.ingest_csv(csv_path)
            print("✅ CSV ingestion completed!")
        except Exception as e:
            print(f"❌ Error during ingestion: {e}")
    else:
        print(f"⚠️  CSV file not found: {csv_path}")
    
    # Example 2: Query the vector database
    print(f"\n🔍 Example 2: Querying vector database")
    try:
        results = rag.query_vector_db("young family with children", n_results=3)
        print(f"✅ Found {len(results)} similar scenarios")
        for i, result in enumerate(results, 1):
            print(f"   Scenario {i}: {result['document'][:100]}...")
    except Exception as e:
        print(f"❌ Error querying database: {e}")
    
    # Example 3: Chat with RAG
    print(f"\n💬 Example 3: Chat with RAG system")
    try:
        response = rag.chat_with_rag("What insurance option would you recommend for a young family?")
        print(f"🤖 AI Response: {response}")
    except Exception as e:
        print(f"❌ Error in RAG chat: {e}")
    
    # Example 4: Get collection information
    print(f"\n📊 Example 4: Collection information")
    try:
        info = rag.get_collection_info()
        print(f"   Collection: {info.get('collection_name')}")
        print(f"   Documents: {info.get('document_count')}")
        print(f"   Directory: {info.get('persist_directory')}")
    except Exception as e:
        print(f"❌ Error getting collection info: {e}")

def example_batch_queries():
    """Example of running multiple queries."""
    
    print("\n" + "="*60)
    print("🔍 Example Batch Queries")
    print("="*60)
    
    rag = RAGPipeline(collection_name="example_insurance")
    
    queries = [
        "What's the best insurance for a single person?",
        "How does age affect insurance recommendations?",
        "What options are available for families with children?",
        "Which insurance is most cost-effective for young professionals?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        try:
            response = rag.chat_with_rag(query)
            print(f"🤖 Response: {response[:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🎯 RAG Pipeline Example Usage")
    print("="*60)
    
    # Run basic examples
    example_usage()
    
    # Run batch query examples
    example_batch_queries()
    
    print("\n✅ Example usage completed!")

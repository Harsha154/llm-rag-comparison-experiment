#!/usr/bin/env python3
"""
Test Script for RAG Pipeline
============================

This script tests the basic functionality of the RAG pipeline.
"""

import os
import sys
from rag_pipeline import RAGPipeline

def test_rag_pipeline():
    """Test the RAG pipeline functionality."""
    print("🧪 Testing RAG Pipeline...")
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline(collection_name="test_insurance", persist_directory="./test_chroma_db")
        print("✅ RAG Pipeline initialized successfully")
        
        # Test collection info
        info = rag.get_collection_info()
        print(f"✅ Collection info retrieved: {info}")
        
        # Test CSV ingestion (if file exists)
        csv_path = "../data/dataset2_to be used in experiments with target --includes ground-truth.csv"
        if os.path.exists(csv_path):
            print(f"📥 Testing CSV ingestion from: {csv_path}")
            rag.ingest_csv(csv_path)
            print("✅ CSV ingestion test completed")
            
            # Test vector database query
            print("🔍 Testing vector database query...")
            results = rag.query_vector_db("young person with children", n_results=3)
            print(f"✅ Query returned {len(results)} results")
            
            # Test RAG chat
            print("💬 Testing RAG chat...")
            response = rag.chat_with_rag("What insurance option would you recommend for a young family?")
            print(f"✅ RAG chat response: {response[:100]}...")
        else:
            print(f"⚠️  CSV file not found: {csv_path}")
            print("   Skipping ingestion and query tests")
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_rag_pipeline()

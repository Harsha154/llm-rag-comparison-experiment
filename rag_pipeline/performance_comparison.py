#!/usr/bin/env python3
"""
Performance Comparison: Original vs Optimized RAG Pipeline
========================================================

This script compares the performance between the original sequential approach
and the new optimized batch processing approach.
"""

import time
import pandas as pd
import tempfile
import os
from rag_pipeline import RAGPipeline
from rag_pipeline_optimized import OptimizedRAGPipeline

def create_test_dataset(num_rows: int = 1000) -> str:
    """Create a test dataset with specified number of rows."""
    print(f"📊 Creating test dataset with {num_rows:,} rows...")
    
    sample_data = []
    for i in range(num_rows):
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
    
    df = pd.DataFrame(sample_data, columns=[
        'Probe ID', 'Age', 'Gender', 'Marital Status', 'Children',
        'Income', 'Health Status', 'Occupation', 'Location',
        'Option 1', 'Option 2', 'Option 3', 'Option 4', 'Ground Truth'
    ])
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_csv = f.name
    
    return temp_csv

def test_original_pipeline(csv_file: str) -> dict:
    """Test the original sequential pipeline."""
    print("\n🔄 Testing Original Pipeline (Sequential Processing)...")
    
    # Initialize original pipeline
    rag = RAGPipeline(collection_name="test_original", persist_directory="./test_chroma_original")
    
    # Measure ingestion time
    start_time = time.time()
    try:
        rag.ingest_csv(csv_file)
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Get document count
        info = rag.get_collection_info()
        doc_count = info.get('document_count', 0)
        
        return {
            'success': True,
            'processing_time': processing_time,
            'documents_processed': doc_count,
            'speed': doc_count / processing_time if processing_time > 0 else 0
        }
    except Exception as e:
        print(f"❌ Error in original pipeline: {e}")
        return {
            'success': False,
            'error': str(e)
        }
    finally:
        # Clean up
        try:
            rag.delete_collection()
        except:
            pass

def test_optimized_pipeline(csv_file: str, batch_size: int = 1000, workers: int = 10) -> dict:
    """Test the optimized pipeline with batch processing."""
    print(f"\n⚡ Testing Optimized Pipeline (Batch Size: {batch_size:,}, Workers: {workers})...")
    
    # Initialize optimized pipeline
    rag = OptimizedRAGPipeline(collection_name="test_optimized", persist_directory="./test_chroma_optimized")
    rag.batch_size = batch_size
    rag.max_workers = workers
    
    # Measure ingestion time
    start_time = time.time()
    try:
        rag.ingest_csv_optimized(csv_file)
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Get document count
        info = rag.get_collection_info()
        doc_count = info.get('document_count', 0)
        
        return {
            'success': True,
            'processing_time': processing_time,
            'documents_processed': doc_count,
            'speed': doc_count / processing_time if processing_time > 0 else 0,
            'batch_size': batch_size,
            'workers': workers
        }
    except Exception as e:
        print(f"❌ Error in optimized pipeline: {e}")
        return {
            'success': False,
            'error': str(e)
        }
    finally:
        # Clean up
        try:
            rag.delete_collection()
        except:
            pass

def compare_performance(test_sizes: list = [100, 500, 1000]):
    """Compare performance across different dataset sizes."""
    print("🚀 RAG Pipeline Performance Comparison")
    print("=" * 60)
    
    results = []
    
    for size in test_sizes:
        print(f"\n📈 Testing with {size:,} documents...")
        
        # Create test dataset
        csv_file = create_test_dataset(size)
        
        try:
            # Test original pipeline
            original_result = test_original_pipeline(csv_file)
            
            # Test optimized pipeline with different configurations
            optimized_configs = [
                {'batch_size': 100, 'workers': 5},
                {'batch_size': 500, 'workers': 10},
                {'batch_size': 1000, 'workers': 10}
            ]
            
            optimized_results = []
            for config in optimized_configs:
                result = test_optimized_pipeline(csv_file, **config)
                optimized_results.append(result)
            
            # Store results
            results.append({
                'dataset_size': size,
                'original': original_result,
                'optimized': optimized_results
            })
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(csv_file)
            except:
                pass
    
    # Print comparison results
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE COMPARISON RESULTS")
    print("=" * 60)
    
    for result in results:
        size = result['dataset_size']
        original = result['original']
        optimized_list = result['optimized']
        
        print(f"\n📈 Dataset Size: {size:,} documents")
        print("-" * 40)
        
        if original['success']:
            print(f"🔄 Original Pipeline:")
            print(f"   Time: {original['processing_time']:.2f} seconds")
            print(f"   Speed: {original['speed']:.2f} docs/sec")
        else:
            print(f"❌ Original Pipeline: Failed - {original.get('error', 'Unknown error')}")
        
        print(f"\n⚡ Optimized Pipeline:")
        for i, opt_result in enumerate(optimized_list):
            if opt_result['success']:
                config = f"Batch {opt_result['batch_size']}, {opt_result['workers']} workers"
                speedup = opt_result['speed'] / original['speed'] if original['success'] and original['speed'] > 0 else 0
                print(f"   {config}:")
                print(f"     Time: {opt_result['processing_time']:.2f} seconds")
                print(f"     Speed: {opt_result['speed']:.2f} docs/sec")
                print(f"     Speedup: {speedup:.1f}x faster")
            else:
                print(f"   ❌ Failed - {opt_result.get('error', 'Unknown error')}")
    
    # Estimate for 69,000 rows
    print(f"\n🎯 ESTIMATED PERFORMANCE FOR 69,000 ROWS")
    print("-" * 40)
    
    if results:
        # Use the largest test size for estimation
        largest_result = results[-1]
        original = largest_result['original']
        best_optimized = max(largest_result['optimized'], key=lambda x: x.get('speed', 0))
        
        if original['success'] and best_optimized['success']:
            original_time_69k = 69000 / original['speed'] / 60  # minutes
            optimized_time_69k = 69000 / best_optimized['speed'] / 60  # minutes
            
            print(f"🔄 Original Pipeline: ~{original_time_69k:.1f} minutes")
            print(f"⚡ Optimized Pipeline: ~{optimized_time_69k:.1f} minutes")
            print(f"🚀 Improvement: {original_time_69k/optimized_time_69k:.1f}x faster")

def main():
    """Main function."""
    print("🎯 RAG Pipeline Performance Comparison")
    print("This will test both original and optimized pipelines with different dataset sizes.")
    print("Note: This will make API calls to OpenAI for embeddings.")
    
    # Test with smaller sizes first
    test_sizes = [100, 500, 1000]
    
    compare_performance(test_sizes)
    
    print(f"\n✅ Performance comparison completed!")
    print(f"💡 For production use with 69,000+ rows, use the optimized pipeline:")

if __name__ == "__main__":
    main()

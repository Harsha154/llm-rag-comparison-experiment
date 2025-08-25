#!/usr/bin/env python3
"""
Test Script for RAG vs Non-RAG Comparison
=========================================

This script runs a small test of the RAG comparison experiment with just a few scenarios
to demonstrate how it works.
"""

from llm_rag_comparison_experiment import RAGComparisonExperiment
import pandas as pd

def test_rag_comparison():
    """Run a small test of the RAG comparison experiment."""
    print("🧪 Testing RAG vs Non-RAG Comparison Experiment")
    print("=" * 80)
    
    # Initialize the experiment
    experiment = RAGComparisonExperiment(
        model="gpt-4",
        temperature=0.7
    )
    
    # Load a small sample of test data
    df = pd.read_csv("data/Test Probes.csv")
    test_sample = df.head(2)  # Just 2 scenarios for testing
    
    print(f"📊 Testing with {len(test_sample)} scenarios")
    print(f"👥 Each scenario will be run for 3 characters: Alex, Brie, Chad")
    print(f"🔄 Each character runs twice: with RAG and without RAG")
    print(f"📈 Total test runs: {len(test_sample) * 3 * 2}")
    print()
    
    # Test each scenario for each character
    for idx, row in test_sample.iterrows():
        print(f"🔍 SCENARIO {idx + 1}:")
        print(f"   Probe ID: {row['id']}")
        print(f"   Insurance Type: {row['probe']}")
        print(f"   Network: {row['network_status']}")
        print(f"   Options: {row['val1']}, {row['val2']}, {row['val3']}, {row['val4']}")
        print()
        
        for character in ['Alex', 'Brie', 'Chad']:
            print(f"   👤 {character} (Risk Aversion: {experiment.characters[character]['risk_aversion']}):")
            
            # Run without RAG
            non_rag_result = experiment.run_single_scenario_character(row, character, use_rag=False)
            print(f"      Non-RAG Choice: {non_rag_result['extracted_choice']}" if non_rag_result['extracted_choice'] else "      Non-RAG Choice: None")
            print(f"      Non-RAG Response: {non_rag_result['llm_output'][:50]}...")
            
            # Run with RAG
            rag_result = experiment.run_single_scenario_character(row, character, use_rag=True)
            print(f"      RAG Choice: {rag_result['extracted_choice']}" if rag_result['extracted_choice'] else "      RAG Choice: None")
            print(f"      RAG Response: {rag_result['llm_output'][:50]}...")
            print()
        
        print("-" * 80)
        print()

if __name__ == "__main__":
    test_rag_comparison()

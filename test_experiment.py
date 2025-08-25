#!/usr/bin/env python3
"""
Test Script for LLM Insurance Experiment
========================================

This script runs a small test of the LLM experiment with just a few scenarios
to demonstrate how it works with character profiles (Alex, Brie, Chad).
"""

from llm_insurance_experiment import InsuranceLLMExperiment
import pandas as pd

def test_experiment():
    """Run a small test of the LLM experiment."""
    print("🧪 Testing LLM Insurance Experiment with Character Profiles")
    print("=" * 60)
    
    # Initialize the experiment
    experiment = InsuranceLLMExperiment(
        model="gpt-4",
        temperature=0.7
    )
    
    # Load a small sample of test data
    df = pd.read_csv("data/Test Probes.csv")
    test_sample = df.head(2)  # Just 2 scenarios for testing
    
    print(f"📊 Testing with {len(test_sample)} scenarios")
    print(f"👥 Each scenario will be run for 3 characters: Alex, Brie, Chad")
    print(f"🔄 Total test runs: {len(test_sample) * 3}")
    print()
    
    # Test each scenario for each character
    for idx, row in test_sample.iterrows():
        print(f"🔍 Scenario {idx + 1}:")
        print(f"   Probe ID: {row['id']}")
        print(f"   Insurance Type: {row['probe']}")
        print(f"   Network: {row['network_status']}")
        print(f"   Options: {row['val1']}, {row['val2']}, {row['val3']}, {row['val4']}")
        print()
        
        for character in ['Alex', 'Brie', 'Chad']:
            print(f"   👤 {character} (Risk Aversion: {experiment.characters[character]['risk_aversion']}):")
            
            # Run the scenario for this character
            result = experiment.run_single_scenario_character(row, character)
            
            print(f"      Choice: {result['llm_choice']}" if result['llm_choice'] else "      Choice: None")
            print(f"      Success: {result['success']}")
            print(f"      Response: {result['llm_response'][:50]}...")
            print()
        
        print("-" * 60)
        print()

if __name__ == "__main__":
    test_experiment()

#!/usr/bin/env python3
"""
Test Script to Print LLM Prompts
================================

This script prints the prompts that would be sent to the LLM for the first 3 scenarios
(9 total prompts: 3 scenarios × 3 characters) without actually calling the API.
"""

from llm_insurance_experiment import InsuranceLLMExperiment
import pandas as pd

def test_prompts():
    """Print the prompts that would be sent to the LLM."""
    print("🧪 Testing LLM Prompts (No API Calls)")
    print("=" * 80)
    
    # Initialize the experiment (no API key needed since we won't make calls)
    experiment = InsuranceLLMExperiment(
        model="gpt-4",
        temperature=0.7
    )
    
    # Load a small sample of test data
    df = pd.read_csv("data/Test Probes.csv")
    test_sample = df.head(3)  # Just 3 scenarios for testing
    
    print(f"📊 Testing with {len(test_sample)} scenarios")
    print(f"👥 Each scenario will be shown for 3 characters: Alex, Brie, Chad")
    print(f"🔄 Total prompts to show: {len(test_sample) * 3}")
    print()
    
    prompt_count = 0
    
    # Test each scenario for each character
    for idx, row in test_sample.iterrows():
        print(f"🔍 SCENARIO {idx + 1}:")
        print(f"   Probe ID: {row['id']}")
        print(f"   Insurance Type: {row['probe']}")
        print(f"   Network: {row['network_status']}")
        print(f"   Expense Type: {row['expense_type']}")
        print(f"   Options: {row['val1']}, {row['val2']}, {row['val3']}, {row['val4']}")
        print()
        
        for character in ['Alex', 'Brie', 'Chad']:
            prompt_count += 1
            print(f"👤 PROMPT {prompt_count} - {character} (Risk Aversion: {experiment.characters[character]['risk_aversion']}):")
            print("-" * 80)
            
            # Format the scenario data
            scenario_data = experiment._format_scenario_data(row)
            
            # Get the choices
            choices = [row['val1'], row['val2'], row['val3'], row['val4']]
            
            # Create the full prompt
            full_prompt = experiment._create_prompt(scenario_data, character, choices)
            
            # Print the prompt
            print(full_prompt)
            print("-" * 80)
            print()
            
            # Break after 9 prompts
            if prompt_count >= 9:
                print(f"🛑 Stopped after {prompt_count} prompts as requested.")
                return
        
        print("=" * 80)
        print()

if __name__ == "__main__":
    test_prompts()

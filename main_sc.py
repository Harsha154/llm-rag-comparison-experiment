#!/usr/bin/env python3
"""
Run 10 test scenarios with Chad, Brie, and Alex as the fixed target characters and save results to CSV.
"""

import json
from self_consistency import SelfConsistencyExperiment


def main():
    exp = SelfConsistencyExperiment(model="gpt-5", temperature=0.7)
    
    # Run pipeline with Chad as target, save to CSV and JSON summary
    print("🚀 Running experiment for Chad...")
    chad_summary = exp.start_pipeline_with_target(
        target_character="Chad",
        csv_path="/Users/harshasureshbabu/Desktop/llm-rag-comparison-experiment/data/Test Probes.csv",
        output_path="chad_10_scenarios_results.csv",
        use_parallel=True,
        max_workers=10)
    
    print("✅ Chad results saved to: chad_10_scenarios_results.csv")
    print("📊 Chad Summary:")
    print(json.dumps(chad_summary, indent=2))
    
    print("\n" + "="*80 + "\n")
    
    # Run pipeline with Brie as target, save to CSV and JSON summary
    print("🚀 Running experiment for Brie...")
    brie_summary = exp.start_pipeline_with_target(
        target_character="Brie",
        csv_path="/Users/harshasureshbabu/Desktop/llm-rag-comparison-experiment/data/Test Probes.csv",
        output_path="brie_10_scenarios_results.csv",
        use_parallel=True,
        max_workers=10)
    
    print("✅ Brie results saved to: brie_10_scenarios_results.csv")
    print("📊 Brie Summary:")
    print(json.dumps(brie_summary, indent=2))
    
    print("\n" + "="*80 + "\n")
    
    # Run pipeline with Alex as target, save to CSV and JSON summary
    print("🚀 Running experiment for Alex...")
    alex_summary = exp.start_pipeline_with_target(
        target_character="Alex",
        csv_path="/Users/harshasureshbabu/Desktop/llm-rag-comparison-experiment/data/Test Probes.csv",
        output_path="alex_10_scenarios_results.csv",
        use_parallel=True,
        max_workers=10)
    
    print("✅ Alex results saved to: alex_10_scenarios_results.csv")
    print("📊 Alex Summary:")
    print(json.dumps(alex_summary, indent=2))


if __name__ == "__main__":
    main()

# from openai import OpenAI
# client = OpenAI()

# response = client.responses.create(
#     model="gpt-5",
#     tools=[{"type": "web_search"}],
#     input="What was a positive news story from today?"
# )

# print(response.output_text)
#!/usr/bin/env python3
"""
Run two example scenarios with Alex as the fixed target character and save results to CSV.
"""

import json
from self_consistency import SelfConsistencyExperiment


def main():
    exp = SelfConsistencyExperiment(model="gpt-4", temperature=0.7)
    
    # Run pipeline with Alex as target, save to CSV and JSON summary
    summary = exp.start_pipeline_with_target(
        target_character="Alex",
        csv_path="/Users/harshasureshbabu/Desktop/llm-rag-comparison-experiment/data/Test Probes.csv",
        output_path="alex_two_scenarios_results.csv",
        max_scenarios=10)
    
    print("✅ Results saved to: alex_two_scenarios_results.csv")
    print("📊 Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()



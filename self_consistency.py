from llm_insurance_experiment import InsuranceLLMExperiment
import pandas as pd
from typing import List, Dict, Any, Optional
import random
from collections import Counter
import argparse
import json

class SelfConsistencyExperiment:
    """Self-consistency experiment that prints prompts for Alex, Chad, and Brie."""
    
    def __init__(self, model: str = "gpt-5", temperature: float = 0.7):
        """
        Initialize the self-consistency experiment.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for model responses
        """
        self.experiment = InsuranceLLMExperiment(model=model, temperature=temperature)
    

    def get_weights(self, target_character: str):
        """
        Return positive weights for the target character. And negative weights for the other characters.
        """
        weights = {}
        for character in self.experiment.characters:
            if character == target_character:
                weights[character] = 1
            else:
                weights[character] = -1
        return weights

    def get_run_jsons(self, target_character: str, scenario_data: str, choices: List[str]):
        """
        Return the run JSONs for the target character and the n assignments.
        """
        weights = self.get_weights(target_character)
        json_input_for_self_consistency = {}
        # Determine the two non-target characters
        all_characters = list(self.experiment.characters.keys())
        non_target_chars = [c for c in all_characters if c != target_character]
        # Randomly assign N values 2 and 3 to the two non-target characters
        n_values = [2, 3]
        random.shuffle(n_values)
        n_assignment: Dict[str, int] = {
            non_target_chars[0]: n_values[0],
            non_target_chars[1]: n_values[1]
        }

        # Build JSON entries for each character
        for character in all_characters:
            if character == target_character:
                json_input_for_self_consistency[character] = {
                    "character": character,
                    "weights": weights[character], #1
                    "prompt": self.experiment._create_prompt(scenario_data, character, choices),
                    "N": 5
                }
            else:
                json_input_for_self_consistency[character] = {
                    "character": character,
                    "weights": weights[character],
                    "prompt": self.experiment._create_prompt(scenario_data, character, choices),
                    "N": n_assignment[character]
                }

        return json_input_for_self_consistency, n_assignment

    def run_self_consistency_for_row(self, row: pd.Series, target_character: str) -> Dict[str, Any]:
        """
        Run self-consistency sampling for a single scenario row given a target character.
        - Target gets N=5 samples (weight=+1)
        - Two non-targets get randomized N in {2,3} (weight=-1)
        Returns per-character samples and weighted vote tallies.
        """
        # Prepare scenario and choices
        scenario_data = self.experiment._format_scenario_data(row)
        choices = [row['val1'], row['val2'], row['val3'], row['val4']]

        # Determine Ns and prompts by character
        run_plan, n_assignment = self.get_run_jsons(target_character=target_character, scenario_data=scenario_data, choices=choices)

        # Execute runs
        per_character_samples: Dict[str, List[str]] = {}
        weighted_votes: Dict[str, int] = {}

        for character, plan in run_plan.items():
            N = plan['N']
            weight = 1 if character == target_character else -1
            samples: List[str] = []

            for _ in range(N):
                result = self.experiment.run_single_scenario_character(row, character)
                extracted = result.get('llm_choice') or result.get('extracted_choice')
                if extracted is not None:
                    samples.append(str(extracted))
                    weighted_votes[str(extracted)] = weighted_votes.get(str(extracted), 0) + weight

            per_character_samples[character] = samples

        # Majority and weighted winners (optional summaries)
        majority_winner: Optional[str] = None
        all_positive_samples = per_character_samples.get(target_character, [])
        if all_positive_samples:
            majority_winner = Counter(all_positive_samples).most_common(1)[0][0]

        weighted_winner: Optional[str] = None
        if weighted_votes:
            weighted_winner = sorted(weighted_votes.items(), key=lambda kv: kv[1], reverse=True)[0][0]

        return {
            'probe_id': row.get('id', None),
            'target_character': target_character,
            'per_character_samples': per_character_samples,
            'majority_winner_positive': majority_winner,
            'weighted_votes': weighted_votes,
            'weighted_winner': weighted_winner,
            'n_assignments': n_assignment
        }

    def run_self_consistency_dataset(self, csv_path: str, target_column: str = 'target_character', max_scenarios: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run self-consistency across a dataset. Each row must include `target_character` (or specify via target_column).
        Returns a list of per-row result dicts.
        """
        df = pd.read_csv(csv_path)
        if max_scenarios is not None:
            df = df.head(max_scenarios)

        results: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            if target_column not in row or pd.isna(row[target_column]):
                # Skip rows without a target definition
                continue
            target_character = str(row[target_column])
            if target_character not in self.experiment.characters:
                # Skip invalid targets
                continue
            results.append(self.run_self_consistency_for_row(row, target_character))

        return results

    def run_self_consistency_dataset_fixed_target(self, csv_path: str, target_character: str, max_scenarios: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run self-consistency across a dataset using a single fixed target character for all rows.
        Ignores any target column in the CSV.
        """
        if target_character not in self.experiment.characters:
            raise ValueError(f"Invalid target_character: {target_character}. Must be one of {list(self.experiment.characters.keys())}")

        df = pd.read_csv(csv_path)
        if max_scenarios is not None:
            df = df.head(max_scenarios)

        results: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            results.append(self.run_self_consistency_for_row(row, target_character))

        return results

    def run_and_save(self,
                     csv_path: str,
                     output_path: str = "self_consistency_results.csv",
                     target_column: str = 'target_character',
                     max_scenarios: Optional[int] = None,
                     fixed_target_character: Optional[str] = None) -> Dict[str, Any]:
        """
        Run weighted self-consistency across a dataset and save results and summary.
        - Writes per-row results to CSV
        - Writes overall summary JSON next to CSV
        Returns a summary dict.
        """
        if fixed_target_character is not None:
            results = self.run_self_consistency_dataset_fixed_target(
                csv_path=csv_path,
                target_character=fixed_target_character,
                max_scenarios=max_scenarios,
            )
        else:
            results = self.run_self_consistency_dataset(
                csv_path=csv_path,
                target_column=target_column,
                max_scenarios=max_scenarios,
            )

        # Flatten results for CSV with new structure
        rows: List[Dict[str, Any]] = []
        for r in results:
            # Get character assignments based on actual n assignments from experiment
            target_character = r.get('target_character')
            per_character_samples = r.get('per_character_samples', {})
            weighted_votes = r.get('weighted_votes', {})
            n_assignments = r.get('n_assignments', {})
            
            # Determine n=2 and n=3 characters from actual assignments
            n2_character = None
            n3_character = None
            for char, n_val in n_assignments.items():
                if n_val == 2:
                    n2_character = char
                elif n_val == 3:
                    n3_character = char
            
            # Extract choices and weights for each character
            target_choices_weights = {}
            n2_choices_weights = {}
            n3_choices_weights = {}
            
            # Get target character's choices and weights
            if target_character in per_character_samples:
                target_samples = per_character_samples[target_character]
                for choice in target_samples:
                    target_choices_weights[choice] = target_choices_weights.get(choice, 0) + 1
            
            # Get n=2 character's choices and weights
            if n2_character and n2_character in per_character_samples:
                n2_samples = per_character_samples[n2_character]
                for choice in n2_samples:
                    n2_choices_weights[choice] = n2_choices_weights.get(choice, 0) + 1
            
            # Get n=3 character's choices and weights
            if n3_character and n3_character in per_character_samples:
                n3_samples = per_character_samples[n3_character]
                for choice in n3_samples:
                    n3_choices_weights[choice] = n3_choices_weights.get(choice, 0) + 1
            
            # Final max weight choice
            final_max_weight_choice = r.get('weighted_winner')
            
            rows.append({
                'probe_id': r.get('probe_id'),
                'target_character': target_character,
                'n=3_character': n3_character,
                'n=2_character': n2_character,
                'target_choices_weights': json.dumps(target_choices_weights),
                'n=2_choices_weights': json.dumps(n2_choices_weights),
                'n=3_choices_weights': json.dumps(n3_choices_weights),
                'final_add_weights_choices': json.dumps(weighted_votes),
                'final_max_weight_choice': final_max_weight_choice,
            })

        df_out = pd.DataFrame(rows)
        df_out.to_csv(output_path, index=False)

        # Build summary
        num_rows = len(results)
        weighted_choice_counts: Dict[str, int] = {}
        for r in results:
            ww = r.get('weighted_winner')
            if ww is not None:
                weighted_choice_counts[ww] = weighted_choice_counts.get(ww, 0) + 1

        summary: Dict[str, Any] = {
            'total_rows_processed': num_rows,
            'weighted_winner_distribution': weighted_choice_counts,
            'model': self.experiment.model,
            'temperature': self.experiment.temperature,
            'characters': self.experiment.characters,
            'input': csv_path,
            'output_csv': output_path,
            'target_column': target_column,
            'max_scenarios': max_scenarios,
            'fixed_target_character': fixed_target_character,
        }

        summary_path = output_path.replace('.csv', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def run_two_example_scenarios_alex(self, csv_path: str = "data/Test Probes.csv") -> List[Dict[str, Any]]:
        """
        Convenience helper: run exactly two scenarios with Alex as the fixed target character.
        Returns the list of per-row result dicts.
        """
        return self.run_self_consistency_dataset_fixed_target(
            csv_path=csv_path,
            target_character='Alex',
            max_scenarios=2,
        )

    def start_pipeline_with_target(self,
                                   target_character: str,
                                   csv_path: str = "data/Test Probes.csv",
                                   output_path: str = "self_consistency_results.csv",
                                   max_scenarios: Optional[int] = None) -> Dict[str, Any]:
        """
        Start the weighted self-consistency pipeline by passing a fixed target character.
        Saves CSV and JSON summary; returns the summary dict.
        """
        return self.run_and_save(
            csv_path=csv_path,
            output_path=output_path,
            max_scenarios=max_scenarios,
            fixed_target_character=target_character,
        )

    def return_character_prompts(self, csv_path: str, scenario_index: int = 0):
        """
        Print the prompts for Alex, Chad, and Brie for a specific scenario.
        
        Args:
            csv_path: Path to the test probes CSV file
            scenario_index: Index of the scenario to use (default: 0)
        """
        # Load the test probes data
        df = pd.read_csv(csv_path)
        
        if scenario_index >= len(df):
            print(f"❌ Scenario index {scenario_index} is out of range. Dataset has {len(df)} scenarios.")
            return
        
        # Get the scenario row
        row = df.iloc[scenario_index]
        
        print(f"🔍 Scenario {scenario_index + 1}: {row['probe']} - {row['network_status']}")
        print(f"📊 Options: A={row['val1']}, B={row['val2']}, C={row['val3']}, D={row['val4']}")
        print("=" * 80)
        
        # Format scenario data (this creates the readable scenario description)
        scenario_data = self.experiment._format_scenario_data(row)
        choices = [row['val1'], row['val2'], row['val3'], row['val4']]
        
        # Print prompts for each character
        characters = ['Alex', 'Brie', 'Chad']

        character_prompt = {}
        
        for character in characters:
            print(f"\n👤 {character} (Risk Aversion: {self.experiment.characters[character]['risk_aversion']})")
            print("-" * 60)
            
            # Create the complete prompt (this includes scenario + character guidance + task)
            prompt = self.experiment._create_prompt(scenario_data, character, choices)
            character_prompt[character] = prompt
            
        return character_prompt
    

def main():
    """CLI entry point for running the weighted self-consistency pipeline."""
    parser = argparse.ArgumentParser(description="Run weighted self-consistency pipeline with character-weighted sampling")
    parser.add_argument("--input", type=str, default="data/Test Probes.csv", help="Path to input CSV")
    parser.add_argument("--output", type=str, default="self_consistency_results.csv", help="Path to output CSV")
    parser.add_argument("--target-column", type=str, default="target_character", help="Column containing target character per row")
    parser.add_argument("--max-scenarios", type=int, default=None, help="Optional limit on number of rows")
    parser.add_argument("--model", type=str, default="gpt5", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--fixed-target-character", type=str, default=None, help="Use a fixed target character for all rows instead of reading from CSV")
    parser.add_argument("--print-prompts", action="store_true", help="Print character prompts for the first scenario and exit")

    args = parser.parse_args()

    experiment = SelfConsistencyExperiment(model=args.model, temperature=args.temperature)

    if args.print_prompts:
        print("🚀 Self-Consistency Experiment - Character Prompt Analysis")
        print("=" * 80)
        experiment.return_character_prompts(args.input, scenario_index=0)
        return

    print("🚀 Running weighted self-consistency pipeline")
    print(f"📥 Input: {args.input}")
    print(f"📤 Output: {args.output}")
    print(f"🎯 Target column: {args.target_column}")
    if args.max_scenarios is not None:
        print(f"📝 Limiting to first {args.max_scenarios} scenarios")

    summary = experiment.run_and_save(
        csv_path=args.input,
        output_path=args.output,
        target_column=args.target_column,
        max_scenarios=args.max_scenarios,
        fixed_target_character=args.fixed_target_character,
    )

    print("\n📈 Summary:")
    print(json.dumps(summary, indent=2))
    

if __name__ == "__main__":
    main()
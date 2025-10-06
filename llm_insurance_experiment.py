#!/usr/bin/env python3
"""
LLM Insurance Decision-Making Experiment
========================================

This script runs an experiment where an LLM is given insurance scenarios from the test probes
and asked to select the most appropriate insurance plan based on the provided prompt.
Each scenario is run three times with different character profiles: Alex, Brie, and Chad.
"""

import os
import pandas as pd
import json
import time
from typing import Dict, List, Any, Optional
from openai import OpenAI
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

# Load environment variables
load_dotenv()

class InsuranceLLMExperiment:
    """Experiment class for running LLM insurance decision-making tests."""
    
    def __init__(self, model: str = "gpt-5", temperature: float = 0.7):
        """
        Initialize the experiment.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for model responses
        """
        self.model = model
        self.temperature = temperature
        self.openai_client = self._initialize_openai()
        
        # Character profiles with risk aversion levels
        self.characters = {
            'Alex': {'risk_aversion': 1.0, 'description': 'highly risk-averse'},
            'Brie': {'risk_aversion': 0.5, 'description': 'moderately risk-tolerant'},
            'Chad': {'risk_aversion': 0.0, 'description': 'highly risk-tolerant'}
        }
        
    def _initialize_openai(self) -> OpenAI:
        """Initialize OpenAI client with credentials."""
        # Try to get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            # Try to get from config file
            config_path = "rag_pipeline/config.yml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    api_key = config.get('openai_api_key')
        
        if not api_key or api_key == "your-openai-api-key-here":
            # Prompt user for API key
            api_key = input("Please enter your OpenAI API key: ").strip()
        
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        return OpenAI(api_key=api_key)
    
    def _get_character_prompt(self, character: str) -> str:
        """Get the character-specific prompt section."""
        if character == 'Chad':
            return """1. The decision-maker profile assigned to you is Chad, and your risk aversion level is 0. You are highly risk-tolerant and will favor high deductibles even at the expense of quality. For each probe in the scenario, the recommended preferences are:

SPECIALIST: Accepts referral requirements and network restrictions for lower costs
URGENT CARE: Uses whatever's cheapest available, unconcerned about provider quality variance
PHARMACY: Chooses highest copays (doesn't think he'll get sick), switches to generics readily
MAIL ORDER: Embraces 90-day bulk despite change/waste risks
OOP MAXIMUM: Accepts highest maximum, betting on staying healthy
TELEMEDICINE: First choice for everything possible to minimize costs
PCP: Skips preventive care if copays apply
OUTPATIENT SURGERY: Chooses lowest-cost facilities/surgeons
Choice influence: LOW - Primarily cost-driven decisions
"""
        
        elif character == 'Brie':
            return """2. The decision-maker profile assigned to you is Brie, and your risk aversion level is 0.5. You are moderately risk-tolerant and aim to balance cost and quality. For each probe in the scenario, the recommended preferences are:
SPECIALIST: Weighs referral hassle against direct access costs
URGENT CARE: Balances quality concerns with convenience needs
PHARMACY: Moderate copays, some brand preference flexibility
MAIL ORDER: Mixes mail and retail based on medication importance
OOP MAXIMUM: Moderate maximum for reasonable protection
TELEMEDICINE: Uses for minor issues, in-person for complex care
PCP: Regular preventive care with moderate copay acceptance
OUTPATIENT SURGERY: Researches options, balances quality and cost
Choice influence: MODERATE - Actively weighing multiple factors"""
        
        elif character == 'Alex':
            return """3. The decision-maker profile assigned to you is Alex, and your risk aversion level is 1. You are highly risk-averse and prioritize predictability and control over cost. For each probe in the scenario, the recommended preferences are:

SPECIALIST: Pays for open access to choose best specialists
URGENT CARE: Prefers known emergency departments despite higher costs
PHARMACY: Lowest copays, maintains brand medications
MAIL ORDER: Avoids for critical meds due to interruption risk
OOP MAXIMUM: Lowest maximum for complete financial protection
TELEMEDICINE: Supplements but doesn't replace in-person care
PCP: Prioritizes continuity with trusted physician
OUTPATIENT SURGERY: Selects top surgeons/facilities regardless of cost
Choice influence: HIGH - Demands maximum control over healthcare decisions
"""
        
        return ""
    
    def _create_prompt(self, scenario_data: str, character: str, choices: List[str]) -> str:
        """Create the complete prompt for a specific character."""
        character_info = self.characters[character]
        risk_aversion = character_info['risk_aversion']
        character_prompt = self._get_character_prompt(character)
        
        # Combine both prompts
        prompt = f"""You are given data about a person who is designated the primary insured and their family, along with details of a medical insurance plan. Each record contains the following features:

probe: The type of insurance plan or coverage item (e.g., OUT-OF-POCKET MAXIMUM, deductible, coinsurance).

network_status: Type of insurance network for the service (e.g., IN-NETWORK, OUT-OF-NETWORK, generic).

expense_type: The type of cost associated with the plan, expressed in dollars or percentage (e.g., COST IN $, copay in $, PERCENT PLAN PAYS).

children_under_4: Number of children in the family under age 4.

children_under_12: Number of children in the family under age 12.

children_under_18: Number of children in the family under age 18.

children_under_26: Number of children in the family under age 26.

employee_type: The payment structure of the primary insured's employment compensation (e.g., salaried, hourly, bonus). Note: salaried employees may have more stable income and benefits.

distance_dm_home_to_employer_hq: Distance from home to primary insured's employer headquarters.

travel_location_known: Whether the person’s work/travel location is known (Yes/No).

owns_rents: Housing status of the person (Owns or Rents).

no_of_medical_visits_previous_year: Number of medical visits the person had in the past year.

percent_family_members_with_chronic_condition: Percentage of family members with chronic medical conditions.

percent_family_members_that_play_sports: Percentage of family members who actively participate in sports.

Task:
Based on these features, you are comparing health insurance plans. Plans differ in upfront costs (premiums) and usage costs (deductibles, copays, coinsurance, out-of-pocket maximums, etc.) You do not know the total cost of premiums, but you may assume that plans with higher out-of-pocket costs are associated with lower premiums, and plans with lower out-of-pocket costs generally have higher premiums. 
When the plan covers a higher percentage of costs, the premium may be higher. 
Each probe presents different types of insurance plan options. Some probes may have absolute costs (e.g., copays, deductibles, out-of-pocket maximums), while others may be framed as percentage values (e.g., coinsurance, percent of costs covered by the plan). You must weigh these differently based on their cost sensitivity, risk aversion, and value placed on predictability versus flexibility.
There are three decision-maker targets with risk aversion levels of 0 (high risk tolerant), 0.5 (moderate risk aversion), and 1 (high risk averse), and your choice must align with the target decision maker's risk aversion level that will be provided to you.
Note: Risk aversion correlates with an implicit attribute called ‘choice,’ which is not directly measured. Low risk aversion corresponds to low choice influence, while high risk aversion corresponds to high choice influence. The maximum level occurs when both risk aversion and choice influence are high.
Your task is to select the insurance plan that aligns the most with the target decision-maker profile assigned to you. Consider how family composition, employment type, health history, and insurance plan characteristics might influence the decision.

Scenario: {scenario_data}

Risk Aversion Profiles:
{character_prompt}

Your Task
For the given probe, your risk aversion level is {risk_aversion} and you have following available options:
A. {choices[0]}  
B. {choices[1]}  
C. {choices[2]}  
D. {choices[3]}  

As {character} with risk aversion = {risk_aversion}, choose the option that best aligns with your decision-maker's profile and risk aversion level. 

Respond with just the value of your chosen option (e.g., 1600, 600, 200, or 0)."""
        
        return prompt
    
    def _format_scenario_data(self, row: pd.Series) -> str:
        """Format a single scenario row into readable text."""
        scenario_parts = []
        
        # Basic scenario info
        scenario_parts.append(f"Insurance Type: {row['probe']}")
        scenario_parts.append(f"Network Status: {row['network_status']}")
        scenario_parts.append(f"Expense Type: {row['expense_type']}")
        
        # Family composition
        total_children = (row['children_under_4'] + row['children_under_12'] + 
                         row['children_under_18'] + row['children_under_26'])
        scenario_parts.append(f"Family Composition: {total_children} total children")
        scenario_parts.append(f"  - Under 4: {row['children_under_4']}")
        scenario_parts.append(f"  - Under 12: {row['children_under_12']}")
        scenario_parts.append(f"  - Under 18: {row['children_under_18']}")
        scenario_parts.append(f"  - Under 26: {row['children_under_26']}")
        
        # Employment and lifestyle
        scenario_parts.append(f"Employment Type: {row['employee_type']}")
        scenario_parts.append(f"Distance to Employer HQ: {row['distance_dm_home_to_employer_hq']} miles")
        scenario_parts.append(f"Travel Location Known: {row['travel_location_known']}")
        scenario_parts.append(f"Housing Status: {row['owns_rents']}")
        
        # Health information
        scenario_parts.append(f"Medical Visits (Previous Year): {row['no_of_medical_visits_previous_year']}")
        scenario_parts.append(f"Family Members with Chronic Conditions: {row['percent_family_members_with_chronic_condition']}%")
        scenario_parts.append(f"Family Members who Play Sports: {row['percent_family_members_that_play_sports']}%")
        
        return "\n".join(scenario_parts)
    
    def _extract_option_choice(self, response: str) -> Optional[str]:
        """Extract the option choice from the LLM response."""
        response_lower = response.lower().strip()
        
        # Look for single letter responses (A, B, C, D) as fallback
        if response_lower in ['a', 'b', 'c', 'd']:
            return response_lower.upper()
        
        # Look for explicit option mentions
        if "option a" in response_lower or "choice a" in response_lower:
            return "A"
        elif "option b" in response_lower or "choice b" in response_lower:
            return "B"
        elif "option c" in response_lower or "choice c" in response_lower:
            return "C"
        elif "option d" in response_lower or "choice d" in response_lower:
            return "D"
        
        # Look for numeric patterns that map to letters
        import re
        option_patterns = [
            r"option\s*(\d)",
            r"choice\s*(\d)",
            r"select\s*(\d)",
            r"recommend\s*(\d)",
            r"(\d)\s*option"
        ]
        
        for pattern in option_patterns:
            matches = re.findall(pattern, response_lower)
            if matches:
                try:
                    choice_num = int(matches[0])
                    if 1 <= choice_num <= 4:
                        return chr(64 + choice_num)  # Convert 1->A, 2->B, 3->C, 4->D
                except ValueError:
                    continue
        
        # Look for actual values (numbers) in the response
        import re
        # Find all numbers in the response
        numbers = re.findall(r'\d+', response)
        if numbers:
            # Convert to integers and return the first one
            try:
                return str(int(numbers[0]))
            except ValueError:
                pass
        
        return None
    
    def run_single_scenario_character(self, row: pd.Series, character: str) -> Dict[str, Any]:
        """Run a single scenario through the LLM for a specific character."""
        try:
            # Format the scenario data
            scenario_data = self._format_scenario_data(row)
            
            # Get the choices
            choices = [row['val1'], row['val2'], row['val3'], row['val4']]
            
            # Create the full prompt
            full_prompt = self._create_prompt(scenario_data, character, choices)
            
            # Get LLM response
            # GPT-5 only supports default temperature (1), so we remove the temperature parameter
            try:
                print(f"🔍 Making API call for {character}...")
                print(f"Model: {self.model}")
                print(f"Prompt length: {len(full_prompt)}")
                
                response = self.openai_client.responses.create(
                    model=self.model,
                    input=full_prompt,
                    # max_completion_tokens=50  # Short response expected
                )
                
                print(f"✅ API response received")
                print(f"Response object: {response}")
                print(f"Choices: {response.output_text}")
                
                llm_response = response.output_text
                print(f"Raw response content: '{llm_response}'")
                
                if not llm_response:
                    llm_response = "No response from LLM"
                    print("⚠️ Empty response detected")
                else:
                    print(f"✅ Got response: '{llm_response}'")
                    
            except Exception as api_error:
                llm_response = f"API Error: {str(api_error)}"
                print(f"❌ API Error: {api_error}")
                import traceback
                traceback.print_exc()
            
            # Extract the option choice
            option_choice = self._extract_option_choice(llm_response)
            
            return {
                'probe_id': row['id'],
                'order_number': row['order number'],
                'probe': row['probe'],
                'network_status': row['network_status'],
                'expense_type': row['expense_type'],
                'character': character,
                'risk_aversion': self.characters[character]['risk_aversion'],
                'option_1': row['val1'],
                'option_2': row['val2'],
                'option_3': row['val3'],
                'option_4': row['val4'],
                'llm_response': llm_response,
                'llm_choice': option_choice,
                'success': option_choice is not None
            }
            
        except Exception as e:
            return {
                'probe_id': row['id'],
                'order_number': row['order number'],
                'probe': row['probe'],
                'network_status': row['network_status'],
                'expense_type': row['expense_type'],
                'character': character,
                'risk_aversion': self.characters[character]['risk_aversion'],
                'option_1': row['val1'],
                'option_2': row['val2'],
                'option_3': row['val3'],
                'option_4': row['val4'],
                'llm_response': f"Error: {str(e)}",
                'llm_choice': None,
                'success': False
            }
    
    def run_experiment(self, csv_path: str, output_path: str, max_scenarios: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the full experiment on the test probes data.
        Each scenario is run three times (once for each character: Alex, Brie, Chad).
        
        Args:
            csv_path: Path to the test probes CSV file
            output_path: Path to save the results
            max_scenarios: Maximum number of scenarios to process (None for all)
        """
        print(f"🚀 Starting LLM Insurance Experiment with Character Profiles")
        print(f"📊 Loading data from: {csv_path}")
        
        # Load the test probes data
        df = pd.read_csv(csv_path)
        
        if max_scenarios:
            df = df.head(max_scenarios)
            print(f"📝 Processing {len(df)} scenarios (limited from {len(pd.read_csv(csv_path))})")
        else:
            print(f"📝 Processing all {len(df)} scenarios")
        
        print(f"👥 Running each scenario for 3 characters: Alex, Brie, Chad")
        print(f"🔄 Total runs: {len(df) * 3}")
        
        results = []
        successful_choices = 0
        total_runs = 0
        
        # Process each scenario for each character
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing scenarios"):
            for character in ['Alex', 'Brie', 'Chad']:
                result = self.run_single_scenario_character(row, character)
                results.append(result)
                total_runs += 1
                
                if result['success']:
                    successful_choices += 1
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.1)
        
        # Calculate statistics
        success_rate = successful_choices / total_runs if total_runs > 0 else 0
        
        # Analyze option distribution by character
        option_counts_by_character = {}
        for character in ['Alex', 'Brie', 'Chad']:
            option_counts_by_character[character] = {}
            for result in results:
                if result['character'] == character and result['llm_choice']:
                    choice = result['llm_choice']
                    option_counts_by_character[character][choice] = option_counts_by_character[character].get(choice, 0) + 1
        
        # Overall option distribution
        overall_option_counts = {}
        for result in results:
            if result['llm_choice']:
                choice = result['llm_choice']
                overall_option_counts[choice] = overall_option_counts.get(choice, 0) + 1
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        
        # Save summary statistics
        summary = {
            'total_scenarios': len(df),
            'total_runs': total_runs,
            'successful_choices': successful_choices,
            'success_rate': success_rate,
            'overall_option_distribution': overall_option_counts,
            'option_distribution_by_character': option_counts_by_character,
            'model': self.model,
            'temperature': self.temperature,
            'characters': self.characters
        }
        
        summary_path = output_path.replace('.csv', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n📈 Experiment Results:")
        print(f"   Total Scenarios: {len(df)}")
        print(f"   Total Runs: {total_runs} (3 per scenario)")
        print(f"   Successful Choices: {successful_choices}")
        print(f"   Success Rate: {success_rate:.2%}")
        print(f"   Overall Option Distribution: {overall_option_counts}")
        print(f"   Results saved to: {output_path}")
        print(f"   Summary saved to: {summary_path}")
        
        # Print character-specific results
        print(f"\n👥 Character-Specific Results:")
        for character in ['Alex', 'Brie', 'Chad']:
            char_results = [r for r in results if r['character'] == character]
            char_success = sum(1 for r in char_results if r['success'])
            char_success_rate = char_success / len(char_results) if char_results else 0
            print(f"   {character} (Risk Aversion: {self.characters[character]['risk_aversion']}):")
            print(f"     Success Rate: {char_success_rate:.2%}")
            print(f"     Option Distribution: {option_counts_by_character[character]}")
        
        return summary

def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description="Run LLM Insurance Decision-Making Experiment with Character Profiles")
    parser.add_argument("--input", type=str, default="data/Test Probes.csv", 
                       help="Path to test probes CSV file")
    parser.add_argument("--output", type=str, default="llm_experiment_results.csv",
                       help="Path to save results")
    parser.add_argument("--model", type=str, default="gpt-5",
                       help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for model responses")
    parser.add_argument("--max-scenarios", type=int, default=None,
                       help="Maximum number of scenarios to process (for testing)")
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = InsuranceLLMExperiment(
        model=args.model,
        temperature=args.temperature
    )
    
    # Run experiment
    summary = experiment.run_experiment(
        csv_path=args.input,
        output_path=args.output,
        max_scenarios=args.max_scenarios
    )
    
    print(f"\n✅ Experiment completed successfully!")

if __name__ == "__main__":
    main()

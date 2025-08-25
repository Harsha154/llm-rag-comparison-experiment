#!/usr/bin/env python3
"""
LLM RAG vs Non-RAG Comparison Experiment
========================================

This script runs the same insurance scenarios with and without RAG pipeline,
comparing how the LLM's decisions differ when given additional context.
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

# Import RAG pipeline
try:
    from rag_pipeline.rag_pipeline import RAGPipeline
    RAG_AVAILABLE = True
except ImportError:
    print("⚠️  RAG pipeline not available. Running without RAG only.")
    RAG_AVAILABLE = False

# Load environment variables
load_dotenv()

class RAGComparisonExperiment:
    """Experiment class for comparing LLM decisions with and without RAG."""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.7, 
                 collection_name: str = "insurance_scenarios", 
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the experiment.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for model responses
            collection_name: ChromaDB collection name for RAG
            persist_directory: ChromaDB persistence directory
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
        
        # Initialize RAG pipeline if available
        self.rag_pipeline = None
        self.rag_available = RAG_AVAILABLE
        if self.rag_available:
            try:
                self.rag_pipeline = RAGPipeline(
                    collection_name=collection_name,
                    persist_directory=persist_directory
                )
                print(f"✅ RAG pipeline initialized with collection: {collection_name}")
            except Exception as e:
                print(f"❌ Failed to initialize RAG pipeline: {e}")
                self.rag_available = False
        
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
            return """1. If your decision-maker profile is Chad, your risk aversion is 0. You are highly risk-tolerant and may favor high deductibles even at the expense of quality. For each probe in the scenario, the recommended preferences are:
SPECIALIST: Accepts referral requirements and network restrictions for lower costs
URGENT CARE: Uses whatever's cheapest available, unconcerned about provider quality variance
PHARMACY: Chooses highest copays (doesn't think he'll get sick), switches to generics readily
MAIL ORDER: Embraces 90-day bulk despite change/waste risks
OOP MAXIMUM: Accepts highest maximum, betting on staying healthy
TELEMEDICINE: First choice for everything possible to minimize costs
PCP: Skips preventive care if copays apply
OUTPATIENT SURGERY: Chooses lowest-cost facilities/surgeons
Choice influence: LOW - Primarily cost-driven decisions"""
        
        elif character == 'Brie':
            return """2. If your decision-maker profile is Brie, your risk aversion is 0.5. You are moderately risk-tolerant and aim to balance cost and quality. For each probe in the scenario, the recommended preferences are:
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
            return """3. If your decision-maker profile is Alex, your risk aversion is 1. You are highly risk-averse and prioritize predictability and control over cost. For each probe in the scenario, the recommended preferences are:
SPECIALIST: Pays for open access to choose best specialists
URGENT CARE: Prefers known emergency departments despite higher costs
PHARMACY: Lowest copays, maintains brand medications
MAIL ORDER: Avoids for critical meds due to interruption risk
OOP MAXIMUM: Lowest maximum for complete financial protection
TELEMEDICINE: Supplements but doesn't replace in-person care
PCP: Prioritizes continuity with trusted physician
OUTPATIENT SURGERY: Selects top surgeons/facilities regardless of cost
Choice influence: HIGH - Demands maximum control over healthcare decisions"""
        
        return ""
    
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
    
    def _get_rag_context(self, scenario_data: str, n_context: int = 3) -> str:
        """Get RAG context for the scenario."""
        if not self.rag_pipeline:
            return ""
        
        try:
            # Query for similar scenarios
            similar_scenarios = self.rag_pipeline.query_vector_db(scenario_data, n_results=n_context)
            
            if not similar_scenarios:
                return ""
            
            # Build context from similar scenarios
            context_parts = []
            for i, scenario in enumerate(similar_scenarios, 1):
                context_parts.append(f"Similar Scenario {i}:\n{scenario['document']}\n")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"⚠️  Error getting RAG context: {e}")
            return ""
    
    def _create_prompt(self, scenario_data: str, character: str, choices: List[str], 
                      use_rag: bool = False, rag_context: str = "") -> str:
        """Create the complete prompt for a specific character."""
        character_info = self.characters[character]
        risk_aversion = character_info['risk_aversion']
        character_prompt = self._get_character_prompt(character)
        
        # Add RAG context if using RAG
        rag_section = ""
        if use_rag and rag_context:
            rag_section = f"""

Additional Context from Similar Insurance Scenarios:
{rag_context}"""
        
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
travel_location_known: Whether the person's work/travel location is known (Yes/No).
owns_rents: Housing status of the person (Owns or Rents).
no_of_medical_visits_previous_year: Number of medical visits the person had in the past year.
percent_family_members_with_chronic_condition: Percentage of family members with chronic medical conditions.
percent_family_members_that_play_sports: Percentage of family members who actively participate in sports.

Scenario: {scenario_data}{rag_section}
You are comparing health insurance plans. Plans differ in upfront costs (premiums) and usage costs (deductibles, copays, coinsurance, out-of-pocket maximums, etc.) 
You do not know the total cost of premiums, but you may assume that plans with higher out-of-pocket costs are associated with lower premiums, and plans with lower out-of-pocket costs generally have higher premiums. 
When the plan covers a higher percentage of costs, the premium may be higher. 
Each probe presents different types of insurance plan options. Some probes may have absolute costs (e.g., copays, deductibles, out-of-pocket maximums), while others may be framed as percentage values (e.g., coinsurance, percent of costs covered by the plan). You must weigh these differently based on their cost sensitivity, risk aversion, and value placed on predictability versus flexibility.
There are three decision-maker profiles based on risk aversion levels and your choice must align to the decision-maker profile that will be provided to you.
Note: Risk aversion correlates with an implicit attribute called 'choice,' which is not directly measured. Low risk aversion corresponds to low choice influence, while high risk aversion corresponds to high choice influence. The maximum level occurs when both risk aversion and choice influence are high.

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
    
    def run_single_scenario_character(self, row: pd.Series, character: str, use_rag: bool = False) -> Dict[str, Any]:
        """Run a single scenario through the LLM for a specific character."""
        try:
            # Format the scenario data
            scenario_data = self._format_scenario_data(row)
            
            # Get the choices
            choices = [row['val1'], row['val2'], row['val3'], row['val4']]
            
            # Get RAG context if using RAG
            rag_context = ""
            if use_rag:
                rag_context = self._get_rag_context(scenario_data)
            
            # Create the full prompt
            full_prompt = self._create_prompt(scenario_data, character, choices, use_rag, rag_context)
            
            # Get LLM response
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are {character}, an insurance decision-maker with specific risk preferences. Respond only with the value of your chosen option (e.g., 1600, 600, 200, or 0)."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.temperature,
                max_tokens=50  # Short response expected
            )
            
            llm_response = response.choices[0].message.content
            
            # Extract the option choice
            option_choice = self._extract_option_choice(llm_response)
            
            return {
                'probe_id': row['id'],
                'character': character,
                'llm_output': llm_response,
                'extracted_choice': option_choice,
                'success': option_choice is not None,
                'use_rag': use_rag
            }
            
        except Exception as e:
            return {
                'probe_id': row['id'],
                'character': character,
                'llm_output': f"Error: {str(e)}",
                'extracted_choice': None,
                'success': False,
                'use_rag': use_rag
            }
    
    def run_comparison_experiment(self, csv_path: str, output_dir: str = "./results", 
                                 max_scenarios: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the comparison experiment with and without RAG.
        
        Args:
            csv_path: Path to the test probes CSV file
            output_dir: Directory to save results
            max_scenarios: Maximum number of scenarios to process (None for all)
        """
        print(f"🚀 Starting LLM RAG vs Non-RAG Comparison Experiment")
        print(f"📊 Loading data from: {csv_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the test probes data
        df = pd.read_csv(csv_path)
        
        if max_scenarios:
            df = df.head(max_scenarios)
            print(f"📝 Processing {len(df)} scenarios (limited from {len(pd.read_csv(csv_path))})")
        else:
            print(f"📝 Processing all {len(df)} scenarios")
        
        print(f"👥 Running each scenario for 3 characters: Alex, Brie, Chad")
        print(f"🔄 Each scenario runs twice: with RAG and without RAG")
        print(f"📈 Total runs: {len(df) * 3 * 2}")
        
        # Initialize results lists
        rag_results = []
        non_rag_results = []
        successful_rag_choices = 0
        successful_non_rag_choices = 0
        total_runs = 0
        
        # Process each scenario for each character
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing scenarios"):
            for character in ['Alex', 'Brie', 'Chad']:
                total_runs += 2  # One with RAG, one without
                
                # Run without RAG
                non_rag_result = self.run_single_scenario_character(row, character, use_rag=False)
                non_rag_results.append(non_rag_result)
                if non_rag_result['success']:
                    successful_non_rag_choices += 1
                
                # Run with RAG
                rag_result = self.run_single_scenario_character(row, character, use_rag=True)
                rag_results.append(rag_result)
                if rag_result['success']:
                    successful_rag_choices += 1
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.1)
        
        # Calculate statistics
        non_rag_success_rate = successful_non_rag_choices / (total_runs // 2) if total_runs > 0 else 0
        rag_success_rate = successful_rag_choices / (total_runs // 2) if total_runs > 0 else 0
        
        # Save results to separate CSV files
        non_rag_df = pd.DataFrame(non_rag_results)
        rag_df = pd.DataFrame(rag_results)
        
        non_rag_output_path = os.path.join(output_dir, "non_rag_results.csv")
        rag_output_path = os.path.join(output_dir, "rag_results.csv")
        
        non_rag_df.to_csv(non_rag_output_path, index=False)
        rag_df.to_csv(rag_output_path, index=False)
        
        # Save summary statistics
        summary = {
            'total_scenarios': len(df),
            'total_runs': total_runs,
            'non_rag_successful_choices': successful_non_rag_choices,
            'rag_successful_choices': successful_rag_choices,
            'non_rag_success_rate': non_rag_success_rate,
            'rag_success_rate': rag_success_rate,
            'model': self.model,
            'temperature': self.temperature,
            'rag_available': self.rag_available
        }
        
        summary_path = os.path.join(output_dir, "comparison_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n📈 Comparison Experiment Results:")
        print(f"   Total Scenarios: {len(df)}")
        print(f"   Total Runs: {total_runs} (2 per scenario × 3 characters)")
        print(f"   Non-RAG Success Rate: {non_rag_success_rate:.2%}")
        print(f"   RAG Success Rate: {rag_success_rate:.2%}")
        print(f"   Non-RAG Results saved to: {non_rag_output_path}")
        print(f"   RAG Results saved to: {rag_output_path}")
        print(f"   Summary saved to: {summary_path}")
        
        return summary

def main():
    """Main function to run the comparison experiment."""
    parser = argparse.ArgumentParser(description="Run LLM RAG vs Non-RAG Comparison Experiment")
    parser.add_argument("--input", type=str, default="data/Test Probes.csv", 
                       help="Path to test probes CSV file")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Directory to save results")
    parser.add_argument("--model", type=str, default="gpt-4",
                       help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for model responses")
    parser.add_argument("--max-scenarios", type=int, default=None,
                       help="Maximum number of scenarios to process (for testing)")
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = RAGComparisonExperiment(
        model=args.model,
        temperature=args.temperature
    )
    
    # Run experiment
    summary = experiment.run_comparison_experiment(
        csv_path=args.input,
        output_dir=args.output_dir,
        max_scenarios=args.max_scenarios
    )
    
    print(f"\n✅ Comparison experiment completed successfully!")

if __name__ == "__main__":
    main()

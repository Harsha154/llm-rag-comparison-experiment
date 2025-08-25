#!/usr/bin/env python3
"""
LLM RAG vs Non-RAG Comparison Experiment (Improved)
==================================================

This script runs the same insurance scenarios with and without RAG pipeline,
comparing how the LLM's decisions differ when given additional context.
Each scenario is run three times with different character profiles: Alex, Brie, and Chad.

IMPROVEMENTS:
- Better rate limiting with exponential backoff
- Resume capability from existing results
- Progress tracking and checkpointing
- Graceful handling of API errors
"""

import os
import pandas as pd
import json
import time
import random
import logging
from typing import Dict, List, Any, Optional, Set
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

class RAGComparisonExperimentImproved:
    """Improved experiment class for comparing LLM decisions with and without RAG."""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.7, 
                 collection_name: str = "insurance_scenarios", 
                 persist_directory: str = "./chroma_db",
                 rate_limit_delay: float = 1.0,
                 max_retries: int = 5,
                 log_file: str = "experiment.log"):
        """
        Initialize the experiment.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for model responses
            collection_name: ChromaDB collection name for RAG
            persist_directory: ChromaDB persistence directory
            rate_limit_delay: Base delay between requests (seconds)
            max_retries: Maximum retries for failed requests
        """
        self.model = model
        self.temperature = temperature
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.log_file = log_file
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
        
        # Setup logging (after rag_available is set)
        self._setup_logging()
        
        if self.rag_available:
            try:
                self.rag_pipeline = RAGPipeline(
                    collection_name=collection_name,
                    persist_directory=persist_directory
                )
                self.logger.info(f"✅ RAG pipeline initialized with collection: {collection_name}")
                print(f"✅ RAG pipeline initialized with collection: {collection_name}")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize RAG pipeline: {e}")
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
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create logger
        self.logger = logging.getLogger('RAGComparisonExperiment')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log experiment start
        self.logger.info("=" * 80)
        self.logger.info("RAG COMPARISON EXPERIMENT STARTED")
        self.logger.info(f"Model: {self.model}")
        self.logger.info(f"Temperature: {self.temperature}")
        self.logger.info(f"Rate Limit Delay: {self.rate_limit_delay}s")
        self.logger.info(f"Max Retries: {self.max_retries}")
        self.logger.info(f"RAG Available: {self.rag_available}")
        self.logger.info("=" * 80)
    
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
                self.logger.debug("No similar scenarios found for RAG context")
                return ""
            
            # Build context from similar scenarios
            context_parts = []
            for i, scenario in enumerate(similar_scenarios, 1):
                context_parts.append(f"Similar Scenario {i}:\n{scenario['document']}\n")
            
            context = "\n".join(context_parts)
            self.logger.debug(f"Retrieved {len(similar_scenarios)} similar scenarios for RAG context")
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting RAG context: {e}")
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
    
    def _make_api_request_with_retry(self, messages: List[Dict[str, str]], probe_id: str, character: str, use_rag: bool) -> Optional[str]:
        """Make API request with exponential backoff and retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Add jitter to avoid thundering herd
                jitter = random.uniform(0.1, 0.5)
                sleep_time = self.rate_limit_delay + jitter
                time.sleep(sleep_time)
                
                self.logger.debug(f"Making API request for {probe_id} - {character} ({'RAG' if use_rag else 'Non-RAG'}) - Attempt {attempt + 1}")
                
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=50
                )
                
                llm_response = response.choices[0].message.content
                
                # Log successful response
                self.logger.info(f"✅ API Success: {probe_id} - {character} ({'RAG' if use_rag else 'Non-RAG'}) - Response: {llm_response}")
                
                return llm_response
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle rate limiting
                if "429" in error_msg or "rate limit" in error_msg:
                    wait_time = (2 ** attempt) * self.rate_limit_delay
                    self.logger.warning(f"⚠️  Rate limited: {probe_id} - {character} ({'RAG' if use_rag else 'Non-RAG'}) - Waiting {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    print(f"⚠️  Rate limited. Waiting {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                
                # Handle other errors
                elif "timeout" in error_msg or "connection" in error_msg:
                    wait_time = (2 ** attempt) * 2
                    self.logger.warning(f"⚠️  Connection error: {probe_id} - {character} ({'RAG' if use_rag else 'Non-RAG'}) - Waiting {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    print(f"⚠️  Connection error. Waiting {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                
                else:
                    self.logger.error(f"❌ API error: {probe_id} - {character} ({'RAG' if use_rag else 'Non-RAG'}) - {e}")
                    print(f"❌ API error: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        self.logger.error(f"❌ API request failed after all retries: {probe_id} - {character} ({'RAG' if use_rag else 'Non-RAG'})")
                        return None
        
        self.logger.error(f"❌ API request failed after all retries: {probe_id} - {character} ({'RAG' if use_rag else 'Non-RAG'})")
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
            
            # Make API request with retry logic
            messages = [
                {"role": "system", "content": f"You are {character}, an insurance decision-maker with specific risk preferences. Respond only with the value of your chosen option (e.g., 1600, 600, 200, or 0)."},
                {"role": "user", "content": full_prompt}
            ]
            
            llm_response = self._make_api_request_with_retry(messages, row['id'], character, use_rag)
            
            if llm_response is None:
                return {
                    'probe_id': row['id'],
                    'character': character,
                    'llm_output': "API request failed after all retries",
                    'extracted_choice': None,
                    'success': False,
                    'use_rag': use_rag
                }
            
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
    
    def _load_existing_results(self, output_dir: str) -> Dict[str, Set[str]]:
        """Load existing results to avoid re-processing."""
        completed = {'non_rag': set(), 'rag': set()}
        
        # Check non-RAG results
        non_rag_path = os.path.join(output_dir, "non_rag_results.csv")
        if os.path.exists(non_rag_path):
            try:
                df = pd.read_csv(non_rag_path)
                for _, row in df.iterrows():
                    key = f"{row['probe_id']}_{row['character']}"
                    completed['non_rag'].add(key)
                self.logger.info(f"📊 Loaded {len(completed['non_rag'])} existing non-RAG results")
                print(f"📊 Loaded {len(completed['non_rag'])} existing non-RAG results")
            except Exception as e:
                self.logger.error(f"⚠️  Error loading non-RAG results: {e}")
                print(f"⚠️  Error loading non-RAG results: {e}")
        
        # Check RAG results
        rag_path = os.path.join(output_dir, "rag_results.csv")
        if os.path.exists(rag_path):
            try:
                df = pd.read_csv(rag_path)
                for _, row in df.iterrows():
                    key = f"{row['probe_id']}_{row['character']}"
                    completed['rag'].add(key)
                self.logger.info(f"📊 Loaded {len(completed['rag'])} existing RAG results")
                print(f"📊 Loaded {len(completed['rag'])} existing RAG results")
            except Exception as e:
                self.logger.error(f"⚠️  Error loading RAG results: {e}")
                print(f"⚠️  Error loading RAG results: {e}")
        
        return completed
    
    def _save_checkpoint(self, output_dir: str, non_rag_results: List[Dict], rag_results: List[Dict]):
        """Save intermediate results as checkpoint."""
        # Save non-RAG results
        non_rag_df = pd.DataFrame(non_rag_results)
        non_rag_path = os.path.join(output_dir, "non_rag_results.csv")
        non_rag_df.to_csv(non_rag_path, index=False)
        
        # Save RAG results
        rag_df = pd.DataFrame(rag_results)
        rag_path = os.path.join(output_dir, "rag_results.csv")
        rag_df.to_csv(rag_path, index=False)
        
        # Save summary
        total_non_rag = len(non_rag_results)
        total_rag = len(rag_results)
        successful_non_rag = sum(1 for r in non_rag_results if r['success'])
        successful_rag = sum(1 for r in rag_results if r['success'])
        
        summary = {
            'total_non_rag_results': total_non_rag,
            'total_rag_results': total_rag,
            'non_rag_successful_choices': successful_non_rag,
            'rag_successful_choices': successful_rag,
            'non_rag_success_rate': successful_non_rag / total_non_rag if total_non_rag > 0 else 0,
            'rag_success_rate': successful_rag / total_rag if total_rag > 0 else 0,
            'model': self.model,
            'temperature': self.temperature,
            'rag_available': self.rag_available,
            'checkpoint_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = os.path.join(output_dir, "comparison_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log checkpoint
        self.logger.info(f"💾 Checkpoint saved: {total_non_rag} non-RAG, {total_rag} RAG results")
        self.logger.info(f"   Non-RAG success rate: {successful_non_rag/total_non_rag:.2%}" if total_non_rag > 0 else "   Non-RAG success rate: 0%")
        self.logger.info(f"   RAG success rate: {successful_rag/total_rag:.2%}" if total_rag > 0 else "   RAG success rate: 0%")
    
    def _save_chunk_results(self, output_dir: str, chunk_num: int, non_rag_results: List[Dict], rag_results: List[Dict]):
        """Save results for a specific chunk."""
        # Create chunk directory
        chunk_dir = os.path.join(output_dir, f"chunk_{chunk_num:03d}")
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Save non-RAG results for this chunk
        if non_rag_results:
            non_rag_df = pd.DataFrame(non_rag_results)
            non_rag_path = os.path.join(chunk_dir, "non_rag_results.csv")
            non_rag_df.to_csv(non_rag_path, index=False)
        
        # Save RAG results for this chunk
        if rag_results:
            rag_df = pd.DataFrame(rag_results)
            rag_path = os.path.join(chunk_dir, "rag_results.csv")
            rag_df.to_csv(rag_path, index=False)
        
        # Save chunk summary
        total_non_rag = len(non_rag_results)
        total_rag = len(rag_results)
        successful_non_rag = sum(1 for r in non_rag_results if r['success'])
        successful_rag = sum(1 for r in rag_results if r['success'])
        
        chunk_summary = {
            'chunk_number': chunk_num,
            'total_non_rag_results': total_non_rag,
            'total_rag_results': total_rag,
            'non_rag_successful_choices': successful_non_rag,
            'rag_successful_choices': successful_rag,
            'non_rag_success_rate': successful_non_rag / total_non_rag if total_non_rag > 0 else 0,
            'rag_success_rate': successful_rag / total_rag if total_rag > 0 else 0,
            'model': self.model,
            'temperature': self.temperature,
            'rag_available': self.rag_available,
            'chunk_completion_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = os.path.join(chunk_dir, "chunk_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(chunk_summary, f, indent=2)
        
        # Log chunk completion
        self.logger.info(f"📦 Chunk {chunk_num} completed: {total_non_rag} non-RAG, {total_rag} RAG results")
        self.logger.info(f"   Non-RAG success rate: {successful_non_rag/total_non_rag:.2%}" if total_non_rag > 0 else "   Non-RAG success rate: 0%")
        self.logger.info(f"   RAG success rate: {successful_rag/total_rag:.2%}" if total_rag > 0 else "   RAG success rate: 0%")
        self.logger.info(f"   Chunk results saved to: {chunk_dir}")
        
        print(f"📦 Chunk {chunk_num} completed: {total_non_rag} non-RAG, {total_rag} RAG results")
        print(f"   Results saved to: {chunk_dir}")
    
    def _combine_all_chunk_results(self, output_dir: str) -> Dict[str, Any]:
        """Combine all chunk results into final summary files."""
        all_non_rag_results = []
        all_rag_results = []
        chunk_summaries = []
        
        # Find all chunk directories
        chunk_dirs = [d for d in os.listdir(output_dir) if d.startswith('chunk_') and os.path.isdir(os.path.join(output_dir, d))]
        chunk_dirs.sort()  # Sort by chunk number
        
        for chunk_dir in chunk_dirs:
            chunk_path = os.path.join(output_dir, chunk_dir)
            
            # Load non-RAG results
            non_rag_path = os.path.join(chunk_path, "non_rag_results.csv")
            if os.path.exists(non_rag_path):
                chunk_non_rag_df = pd.read_csv(non_rag_path)
                all_non_rag_results.extend(chunk_non_rag_df.to_dict('records'))
            
            # Load RAG results
            rag_path = os.path.join(chunk_path, "rag_results.csv")
            if os.path.exists(rag_path):
                chunk_rag_df = pd.read_csv(rag_path)
                all_rag_results.extend(chunk_rag_df.to_dict('records'))
            
            # Load chunk summary
            summary_path = os.path.join(chunk_path, "chunk_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    chunk_summary = json.load(f)
                    chunk_summaries.append(chunk_summary)
        
        # Save combined results
        if all_non_rag_results:
            combined_non_rag_df = pd.DataFrame(all_non_rag_results)
            combined_non_rag_path = os.path.join(output_dir, "combined_non_rag_results.csv")
            combined_non_rag_df.to_csv(combined_non_rag_path, index=False)
        
        if all_rag_results:
            combined_rag_df = pd.DataFrame(all_rag_results)
            combined_rag_path = os.path.join(output_dir, "combined_rag_results.csv")
            combined_rag_df.to_csv(combined_rag_path, index=False)
        
        # Calculate overall statistics
        total_non_rag = len(all_non_rag_results)
        total_rag = len(all_rag_results)
        successful_non_rag = sum(1 for r in all_non_rag_results if r['success'])
        successful_rag = sum(1 for r in all_rag_results if r['success'])
        
        overall_summary = {
            'total_chunks': len(chunk_dirs),
            'total_non_rag_results': total_non_rag,
            'total_rag_results': total_rag,
            'non_rag_successful_choices': successful_non_rag,
            'rag_successful_choices': successful_rag,
            'non_rag_success_rate': successful_non_rag / total_non_rag if total_non_rag > 0 else 0,
            'rag_success_rate': successful_rag / total_rag if total_rag > 0 else 0,
            'model': self.model,
            'temperature': self.temperature,
            'rag_available': self.rag_available,
            'chunk_summaries': chunk_summaries,
            'completion_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save overall summary
        overall_summary_path = os.path.join(output_dir, "overall_summary.json")
        with open(overall_summary_path, 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        self.logger.info(f"📊 Combined {len(chunk_dirs)} chunks: {total_non_rag} non-RAG, {total_rag} RAG results")
        print(f"📊 Combined {len(chunk_dirs)} chunks: {total_non_rag} non-RAG, {total_rag} RAG results")
        
        return overall_summary
    
    def run_comparison_experiment(self, csv_path: str, output_dir: str = "./results", 
                                 max_scenarios: Optional[int] = None,
                                 resume: bool = True,
                                 chunk_size: int = 500) -> Dict[str, Any]:
        """
        Run the comparison experiment with and without RAG.
        
        Args:
            csv_path: Path to the test probes CSV file
            output_dir: Directory to save results
            max_scenarios: Maximum number of scenarios to process (None for all)
            resume: Whether to resume from existing results
            chunk_size: Number of scenarios to process per chunk
        """
        self.logger.info("🚀 Starting LLM RAG vs Non-RAG Comparison Experiment (Improved)")
        self.logger.info(f"📊 Loading data from: {csv_path}")
        print(f"🚀 Starting LLM RAG vs Non-RAG Comparison Experiment (Improved)")
        print(f"📊 Loading data from: {csv_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the test probes data
        df = pd.read_csv(csv_path)
        
        if max_scenarios:
            df = df.head(max_scenarios)
            self.logger.info(f"📝 Processing {len(df)} scenarios (limited from {len(pd.read_csv(csv_path))})")
            print(f"📝 Processing {len(df)} scenarios (limited from {len(pd.read_csv(csv_path))})")
        else:
            self.logger.info(f"📝 Processing all {len(df)} scenarios")
            print(f"📝 Processing all {len(df)} scenarios")
        
        self.logger.info(f"👥 Running each scenario for 3 characters: Alex, Brie, Chad")
        self.logger.info(f"🔄 Each scenario runs twice: with RAG and without RAG")
        self.logger.info(f"📈 Total runs: {len(df) * 3 * 2}")
        self.logger.info(f"⏱️  Rate limit delay: {self.rate_limit_delay}s between requests")
        self.logger.info(f"🔄 Max retries: {self.max_retries}")
        self.logger.info(f"📦 Chunk size: {chunk_size} scenarios per chunk")
        print(f"👥 Running each scenario for 3 characters: Alex, Brie, Chad")
        print(f"🔄 Each scenario runs twice: with RAG and without RAG")
        print(f"📈 Total runs: {len(df) * 3 * 2}")
        print(f"⏱️  Rate limit delay: {self.rate_limit_delay}s between requests")
        print(f"🔄 Max retries: {self.max_retries}")
        print(f"📦 Chunk size: {chunk_size} scenarios per chunk")
        
        # Load existing results if resuming
        completed = {'non_rag': set(), 'rag': set()}
        if resume:
            completed = self._load_existing_results(output_dir)
        
        # Initialize results lists
        non_rag_results = []
        rag_results = []
        
        # Load existing results into lists
        if resume:
            non_rag_path = os.path.join(output_dir, "non_rag_results.csv")
            rag_path = os.path.join(output_dir, "rag_results.csv")
            
            if os.path.exists(non_rag_path):
                non_rag_df = pd.read_csv(non_rag_path)
                non_rag_results = non_rag_df.to_dict('records')
            
            if os.path.exists(rag_path):
                rag_df = pd.read_csv(rag_path)
                rag_results = rag_df.to_dict('records')
        
        # Process scenarios in chunks
        total_scenarios = len(df)
        num_chunks = (total_scenarios + chunk_size - 1) // chunk_size  # Ceiling division
        
        self.logger.info(f"📦 Processing {total_scenarios} scenarios in {num_chunks} chunks of {chunk_size}")
        print(f"📦 Processing {total_scenarios} scenarios in {num_chunks} chunks of {chunk_size}")
        
        all_non_rag_results = []
        all_rag_results = []
        
        for chunk_num in range(num_chunks):
            start_idx = chunk_num * chunk_size
            end_idx = min((chunk_num + 1) * chunk_size, total_scenarios)
            chunk_df = df.iloc[start_idx:end_idx]
            
            self.logger.info(f"📦 Starting chunk {chunk_num + 1}/{num_chunks}: scenarios {start_idx + 1}-{end_idx}")
            print(f"\n📦 Starting chunk {chunk_num + 1}/{num_chunks}: scenarios {start_idx + 1}-{end_idx}")
            
            # Initialize chunk results
            chunk_non_rag_results = []
            chunk_rag_results = []
            
            # Process each scenario in this chunk
            for idx, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc=f"Chunk {chunk_num + 1}"):
                # Log scenario start
                self.logger.info(f"🔍 Processing scenario {idx + 1}: {row['id']} - {row['probe']}")
                
                for character in ['Alex', 'Brie', 'Chad']:
                    # Check if non-RAG result already exists
                    non_rag_key = f"{row['id']}_{character}"
                    if non_rag_key not in completed['non_rag']:
                        self.logger.info(f"  📝 Running non-RAG for {character}")
                        non_rag_result = self.run_single_scenario_character(row, character, use_rag=False)
                        chunk_non_rag_results.append(non_rag_result)
                        all_non_rag_results.append(non_rag_result)
                        completed['non_rag'].add(non_rag_key)
                    else:
                        self.logger.debug(f"  ⏭️  Skipping non-RAG for {character} (already completed)")
                    
                    # Check if RAG result already exists
                    rag_key = f"{row['id']}_{character}"
                    if rag_key not in completed['rag']:
                        self.logger.info(f"  📝 Running RAG for {character}")
                        rag_result = self.run_single_scenario_character(row, character, use_rag=True)
                        chunk_rag_results.append(rag_result)
                        all_rag_results.append(rag_result)
                        completed['rag'].add(rag_key)
                    else:
                        self.logger.debug(f"  ⏭️  Skipping RAG for {character} (already completed)")
            
            # Save chunk results
            self.logger.info(f"💾 Saving chunk {chunk_num + 1} results...")
            self._save_chunk_results(output_dir, chunk_num + 1, chunk_non_rag_results, chunk_rag_results)
        
        # Combine all chunk results
        self.logger.info("📊 Combining all chunk results...")
        print(f"\n📊 Combining all chunk results...")
        overall_summary = self._combine_all_chunk_results(output_dir)
        
        # Log and print final summary
        self.logger.info("=" * 80)
        self.logger.info("📈 COMPARISON EXPERIMENT RESULTS")
        self.logger.info(f"   Total Scenarios Processed: {total_scenarios}")
        self.logger.info(f"   Total Chunks: {num_chunks}")
        self.logger.info(f"   Non-RAG Results: {overall_summary['total_non_rag_results']}")
        self.logger.info(f"   RAG Results: {overall_summary['total_rag_results']}")
        self.logger.info(f"   Non-RAG Success Rate: {overall_summary['non_rag_success_rate']:.2%}")
        self.logger.info(f"   RAG Success Rate: {overall_summary['rag_success_rate']:.2%}")
        self.logger.info(f"   Combined Results saved to: {os.path.join(output_dir, 'combined_non_rag_results.csv')}")
        self.logger.info(f"   Combined Results saved to: {os.path.join(output_dir, 'combined_rag_results.csv')}")
        self.logger.info(f"   Overall Summary saved to: {os.path.join(output_dir, 'overall_summary.json')}")
        self.logger.info(f"   Log file: {self.log_file}")
        self.logger.info("=" * 80)
        
        print(f"\n📈 Comparison Experiment Results:")
        print(f"   Total Scenarios Processed: {total_scenarios}")
        print(f"   Total Chunks: {num_chunks}")
        print(f"   Non-RAG Results: {overall_summary['total_non_rag_results']}")
        print(f"   RAG Results: {overall_summary['total_rag_results']}")
        print(f"   Non-RAG Success Rate: {overall_summary['non_rag_success_rate']:.2%}")
        print(f"   RAG Success Rate: {overall_summary['rag_success_rate']:.2%}")
        print(f"   Combined Results saved to: {os.path.join(output_dir, 'combined_non_rag_results.csv')}")
        print(f"   Combined Results saved to: {os.path.join(output_dir, 'combined_rag_results.csv')}")
        print(f"   Overall Summary saved to: {os.path.join(output_dir, 'overall_summary.json')}")
        print(f"   Log file: {self.log_file}")
        
        return overall_summary

def main():
    """Main function to run the comparison experiment."""
    parser = argparse.ArgumentParser(description="Run LLM RAG vs Non-RAG Comparison Experiment (Improved)")
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
    parser.add_argument("--rate-limit-delay", type=float, default=2.0,
                       help="Base delay between API requests (seconds)")
    parser.add_argument("--max-retries", type=int, default=5,
                       help="Maximum retries for failed requests")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from existing results")
    parser.add_argument("--log-file", type=str, default="experiment.log",
                       help="Log file path")
    parser.add_argument("--chunk-size", type=int, default=500,
                       help="Number of scenarios to process per chunk")
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = RAGComparisonExperimentImproved(
        model=args.model,
        temperature=args.temperature,
        rate_limit_delay=args.rate_limit_delay,
        max_retries=args.max_retries,
        log_file=args.log_file
    )
    
    # Run experiment
    summary = experiment.run_comparison_experiment(
        csv_path=args.input,
        output_dir=args.output_dir,
        max_scenarios=args.max_scenarios,
        resume=not args.no_resume,
        chunk_size=args.chunk_size
    )
    
    print(f"\n✅ Comparison experiment completed successfully!")

if __name__ == "__main__":
    main()

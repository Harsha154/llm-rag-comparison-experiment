# LLM Insurance Decision-Making Experiment with Character Profiles

This experiment tests how well an LLM can make insurance decisions based on different risk aversion profiles using the test probes dataset. Each scenario is run three times with different character profiles: Alex, Brie, and Chad.

## 🎯 **Experiment Overview**

The experiment presents insurance scenarios to an LLM and asks it to select the most appropriate insurance plan from 4 options based on different risk aversion profiles:

### **Character Profiles:**
- **Alex** (Risk Aversion: 1.0): Highly risk-averse, prioritizes predictability and control over cost
- **Brie** (Risk Aversion: 0.5): Moderately risk-tolerant, balances cost and quality
- **Chad** (Risk Aversion: 0.0): Highly risk-tolerant, primarily cost-driven decisions

Each scenario is run three times (once for each character) to test how different risk preferences influence insurance decisions.

## 📁 **Files**

- `llm_insurance_experiment.py` - Main experiment script with character profiles
- `test_experiment.py` - Test script for small-scale testing
- `data/Test Probes.csv` - Test scenarios (1,002 scenarios)
- `data/test_ground_truth_only.csv` - Ground truth data for evaluation

## 🚀 **Quick Start**

### 1. **Test with a few scenarios** (Recommended first step)
```bash
python test_experiment.py
```

### 2. **Run full experiment**
```bash
python llm_insurance_experiment.py --input data/Test Probes.csv --output results.csv
```

### 3. **Run with limited scenarios** (for testing)
```bash
python llm_insurance_experiment.py --max-scenarios 10 --output test_results.csv
```

## ⚙️ **Configuration Options**

### Command Line Arguments
- `--input`: Path to test probes CSV file (default: `data/Test Probes.csv`)
- `--output`: Path to save results (default: `llm_experiment_results.csv`)
- `--model`: OpenAI model to use (default: `gpt-4`)
- `--temperature`: Model temperature (default: `0.7`)
- `--max-scenarios`: Limit number of scenarios (default: all)

### Examples
```bash
# Use GPT-3.5-turbo with higher temperature
python llm_insurance_experiment.py --model gpt-3.5-turbo --temperature 0.9

# Process only first 50 scenarios
python llm_insurance_experiment.py --max-scenarios 50 --output small_test.csv

# Custom input/output paths
python llm_insurance_experiment.py --input my_data.csv --output my_results.csv
```

## 📊 **Output Files**

### 1. **Results CSV** (`results.csv`)
Contains detailed results for each scenario and character:
- Original scenario data
- Character profile (Alex, Brie, or Chad)
- Risk aversion level
- LLM response text
- Extracted option choice (A, B, C, or D)
- Success flag (whether choice was extracted)

### 2. **Summary JSON** (`results_summary.json`)
Contains experiment statistics:
- Total scenarios processed
- Total runs (scenarios × 3 characters)
- Success rate
- Overall option distribution
- Option distribution by character
- Model and temperature used

## 🔍 **Understanding the Data**

### Scenario Structure
Each scenario contains:
- **Insurance Type**: Deductible, out-of-pocket maximum, specialist visits, etc.
- **Network Status**: In-network, out-of-network, tier 1, generic
- **Expense Type**: Cost in $, co-pay, percentage plan pays
- **Family Demographics**: Children by age groups
- **Employment**: Salaried, hourly, bonus-based
- **Health Factors**: Medical visits, chronic conditions, sports participation
- **Lifestyle**: Housing status, travel patterns, distance to work

### Insurance Options
Each scenario presents 4 options (val1, val2, val3, val4) with different:
- Cost levels
- Coverage restrictions
- Network limitations
- Risk/benefit trade-offs

## 🧪 **Experiment Design**

### Character Profiles

#### **Alex (Risk Aversion: 1.0)**
- **Profile**: Highly risk-averse, prioritizes predictability and control
- **Preferences**:
  - SPECIALIST: Pays for open access to choose best specialists
  - URGENT CARE: Prefers known emergency departments despite higher costs
  - PHARMACY: Lowest copays, maintains brand medications
  - MAIL ORDER: Avoids for critical meds due to interruption risk
  - OOP MAXIMUM: Lowest maximum for complete financial protection
  - TELEMEDICINE: Supplements but doesn't replace in-person care
  - PCP: Prioritizes continuity with trusted physician
  - OUTPATIENT SURGERY: Selects top surgeons/facilities regardless of cost
- **Choice Influence**: HIGH - Demands maximum control over healthcare decisions

#### **Brie (Risk Aversion: 0.5)**
- **Profile**: Moderately risk-tolerant, balances cost and quality
- **Preferences**:
  - SPECIALIST: Weighs referral hassle against direct access costs
  - URGENT CARE: Balances quality concerns with convenience needs
  - PHARMACY: Moderate copays, some brand preference flexibility
  - MAIL ORDER: Mixes mail and retail based on medication importance
  - OOP MAXIMUM: Moderate maximum for reasonable protection
  - TELEMEDICINE: Uses for minor issues, in-person for complex care
  - PCP: Regular preventive care with moderate copay acceptance
  - OUTPATIENT SURGERY: Researches options, balances quality and cost
- **Choice Influence**: MODERATE - Actively weighing multiple factors

#### **Chad (Risk Aversion: 0.0)**
- **Profile**: Highly risk-tolerant, primarily cost-driven
- **Preferences**:
  - SPECIALIST: Accepts referral requirements and network restrictions for lower costs
  - URGENT CARE: Uses whatever's cheapest available, unconcerned about provider quality
  - PHARMACY: Chooses highest copays, switches to generics readily
  - MAIL ORDER: Embraces 90-day bulk despite change/waste risks
  - OOP MAXIMUM: Accepts highest maximum, betting on staying healthy
  - TELEMEDICINE: First choice for everything possible to minimize costs
  - PCP: Skips preventive care if copays apply
  - OUTPATIENT SURGERY: Chooses lowest-cost facilities/surgeons
- **Choice Influence**: LOW - Primarily cost-driven decisions

### Prompt Structure
The LLM receives:
1. **Context**: Explanation of all features and their meaning
2. **Scenario Data**: Formatted information about the person/family
3. **Insurance Context**: Information about premium vs. out-of-pocket cost trade-offs
4. **Character Profile**: Specific risk aversion profile and preferences
5. **Task**: Choose option A, B, C, or D based on character profile
6. **Instructions**: Respond with just the letter choice

### Response Processing
The script extracts the LLM's choice using pattern matching:
- Actual values (numbers like 1600, 600, 200, 0)
- Single letter responses (A, B, C, D) as fallback
- Explicit option mentions ("Option A", "Choice B", etc.)
- Numeric patterns that map to letters (1→A, 2→B, 3→C, 4→D)
- Fallback to None if no clear choice is found

## 📈 **Expected Results**

### Success Metrics
- **Success Rate**: Percentage of scenarios where LLM made a clear choice
- **Option Distribution**: How often each option was selected overall
- **Character-Specific Patterns**: How each character's choices differ
- **Response Quality**: Analysis of reasoning and decision factors

### Analysis Opportunities
- Compare LLM choices with human decisions (using ground truth data)
- Analyze decision patterns by character and risk aversion level
- Study how different factors influence LLM choices across characters
- Evaluate consistency within each character profile
- Test if LLMs show expected risk aversion patterns

## 🔧 **Setup Requirements**

### Dependencies
```bash
pip install openai pandas tqdm python-dotenv pyyaml
```

### OpenAI API Key
Set your OpenAI API key in one of these ways:
1. Environment variable: `export OPENAI_API_KEY="your-key"`
2. Config file: Edit `rag_pipeline/config.yml`
3. Interactive prompt: Enter when script runs

## 🚨 **Important Notes**

### Rate Limiting
- The script includes a 0.1-second delay between requests
- For large datasets, consider using the `--max-scenarios` flag
- Monitor your OpenAI API usage and costs
- **Note**: Each scenario runs 3 times (once per character), so total API calls = scenarios × 3

### Data Privacy
- All scenario data is sent to OpenAI's API
- Consider data privacy implications for sensitive information
- Results are saved locally and not shared

### Model Selection
- `gpt-4`: Best reasoning, higher cost
- `gpt-3.5-turbo`: Faster, lower cost, may have lower accuracy
- Test with small samples first to find optimal settings

## 🔬 **Research Applications**

This experiment can be used to:
- **Compare LLM vs Human Decision-Making**: Analyze how well LLMs align with human insurance choices
- **Study Risk Aversion Patterns**: Test if LLMs show expected risk aversion behavior
- **Evaluate Character Consistency**: Test if LLMs maintain consistent character profiles
- **Analyze Decision Factors**: Understand which factors most influence LLM decisions
- **Bias Analysis**: Examine if LLMs show demographic or other biases in decisions
- **Prompt Engineering**: Test different prompt formulations for better decision-making
- **Cross-Character Analysis**: Compare how the same scenario is handled by different risk profiles

## 📝 **Example Usage Workflow**

1. **Start Small**: Run `test_experiment.py` to verify setup
2. **Test Parameters**: Try different models and temperatures with limited scenarios
3. **Full Experiment**: Run complete experiment with your chosen parameters
4. **Analyze Results**: Examine the CSV and JSON output files
5. **Compare with Ground Truth**: Use `test_ground_truth_only.csv` for evaluation
6. **Character Analysis**: Compare decision patterns across Alex, Brie, and Chad
7. **Iterate**: Adjust prompts, models, or parameters based on results

## 📊 **Sample Output**

### Results CSV Structure:
```
probe_id,order_number,probe,network_status,expense_type,character,risk_aversion,option_1,option_2,option_3,option_4,llm_response,llm_choice,success
probe_10005_...,10005,DEDUCTIBLE,IN-NETWORK,COST IN $,Alex,1.0,1600,600,200,0,1600,True
probe_10005_...,10005,DEDUCTIBLE,IN-NETWORK,COST IN $,Brie,0.5,1600,600,200,0,600,True
probe_10005_...,10005,DEDUCTIBLE,IN-NETWORK,COST IN $,Chad,0.0,1600,600,200,0,0,True
```

### Summary JSON Structure:
```json
{
  "total_scenarios": 1002,
  "total_runs": 3006,
  "successful_choices": 2850,
  "success_rate": 0.948,
  "overall_option_distribution": {"1600": 750, "600": 800, "200": 700, "0": 600},
  "option_distribution_by_character": {
    "Alex": {"1600": 300, "600": 200, "200": 150, "0": 100},
    "Brie": {"1600": 250, "600": 300, "200": 250, "0": 200},
    "Chad": {"1600": 200, "600": 300, "200": 300, "0": 300}
  }
}
```

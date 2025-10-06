# LLM RAG vs Non-RAG Comparison Experiment

This experiment compares how LLM insurance decisions differ when using RAG (Retrieval-Augmented Generation) versus standard prompting. Each scenario is run twice - once with RAG context and once without - for all three character profiles (Alex, Brie, Chad).

## 🎯 **Experiment Overview**

The experiment runs the same insurance scenarios with and without RAG pipeline to compare:
- **Decision consistency** between RAG and non-RAG approaches
- **Impact of additional context** on LLM choices
- **Character profile adherence** with and without RAG
- **Response quality** and reasoning differences

### **Workflow:**
1. **Load test scenarios** from `data/Test Probes.csv`
2. **For each scenario and character**:
   - Run **without RAG**: Standard prompt only
   - Run **with RAG**: Standard prompt + similar scenarios context
3. **Save results** in separate CSV files
4. **Compare outcomes** between RAG and non-RAG approaches

## 📁 **Files**

- `llm_rag_comparison_experiment.py` - Main comparison experiment script
- `test_rag_comparison.py` - Test script for small-scale testing
- `data/Test Probes.csv` - Test scenarios (1,002 scenarios)
- `rag_pipeline/` - RAG pipeline implementation (must be set up first)

## 🚀 **Quick Start**

### **Prerequisites:**
1. **Set up RAG pipeline** (if not already done):
   ```bash
   cd rag_pipeline
   python cli.py --ingest "../data/training-data-RAG.csv"
   ```

### **1. Test with a few scenarios** (Recommended first step)
```bash
python test_rag_comparison.py
```

### **2. Run full comparison experiment**
```bash
python llm_rag_comparison_experiment.py --input data/Test Probes.csv --output-dir ./results
```

### **3. Run with limited scenarios** (for testing)
```bash
python llm_rag_comparison_experiment.py --max-scenarios 10 --output-dir ./test_results
```

## ⚙️ **Configuration Options**

### Command Line Arguments
- `--input`: Path to test probes CSV file (default: `data/Test Probes.csv`)
- `--output-dir`: Directory to save results (default: `./results`)
- `--model`: OpenAI model to use (default: `gpt-4`)
- `--temperature`: Model temperature (default: `0.7`)
- `--max-scenarios`: Limit number of scenarios (default: all)

### Examples
```bash
# Use GPT-3.5-turbo with higher temperature
python llm_rag_comparison_experiment.py --model gpt-3.5-turbo --temperature 0.9

# Process only first 50 scenarios
python llm_rag_comparison_experiment.py --max-scenarios 50 --output-dir small_test

# Custom input/output paths
python llm_rag_comparison_experiment.py --input my_data.csv --output-dir my_results
```

## 📊 **Output Files**

### **1. Non-RAG Results** (`non_rag_results.csv`)
Contains results for scenarios run without RAG context:
- `probe_id`: Unique scenario identifier
- `character`: Character profile (Alex, Brie, or Chad)
- `llm_output`: Full LLM response text
- `extracted_choice`: Parsed choice value
- `success`: Whether choice was successfully extracted
- `use_rag`: Always False for this file

### **2. RAG Results** (`rag_results.csv`)
Contains results for scenarios run with RAG context:
- Same columns as non-RAG results
- `use_rag`: Always True for this file
- `llm_output`: May include reasoning based on similar scenarios

### **3. Summary JSON** (`comparison_summary.json`)
Contains experiment statistics:
- Total scenarios and runs
- Success rates for both approaches
- Model and temperature used
- RAG availability status

## 🔍 **Understanding the Comparison**

### **RAG Context Addition**
When using RAG, the LLM receives additional context:
```
Additional Context from Similar Insurance Scenarios:
Similar Scenario 1:
Insurance Type: DEDUCTIBLE | Network Status: IN-NETWORK | Family Composition: 2 total children...

Similar Scenario 2:
Insurance Type: DEDUCTIBLE | Network Status: OUT-OF-NETWORK | Family Composition: 1 total children...

Similar Scenario 3:
Insurance Type: DEDUCTIBLE | Network Status: TIER 1 NETWORK | Family Composition: 3 total children...
```

### **Expected Differences**
- **Non-RAG**: Decisions based on character profile and scenario data only
- **RAG**: Decisions influenced by similar historical scenarios and patterns

## 🧪 **Experiment Design**

### **Character Profiles**
Same three profiles as before:
- **Alex** (Risk Aversion: 1.0): Highly risk-averse
- **Brie** (Risk Aversion: 0.5): Moderately risk-tolerant  
- **Chad** (Risk Aversion: 0.0): Highly risk-tolerant

### **Prompt Structure**
Both approaches use the same base prompt, but RAG adds:
- Similar scenario context from training data
- Historical decision patterns
- Additional reasoning context

### **Response Processing**
Same extraction logic for both approaches:
- Extract actual values (1600, 600, 200, 0)
- Fallback to letter choices (A, B, C, D)
- Track success rates for comparison

## 📈 **Analysis Opportunities**

### **Direct Comparisons**
- **Choice Consistency**: Do characters make the same choices with/without RAG?
- **Decision Patterns**: How does RAG influence decision-making?
- **Success Rates**: Are responses more reliable with additional context?

### **Character-Specific Analysis**
- **Alex**: Does RAG help maintain risk-averse choices?
- **Brie**: Does RAG improve balanced decision-making?
- **Chad**: Does RAG reinforce cost-driven choices?

### **Scenario-Specific Analysis**
- **Complex Scenarios**: Does RAG help with difficult decisions?
- **Edge Cases**: How does RAG handle unusual scenarios?
- **Pattern Recognition**: Does RAG help identify decision patterns?

## 🔧 **Setup Requirements**

### **Dependencies**
```bash
pip install openai pandas tqdm python-dotenv pyyaml chromadb
```

### **RAG Pipeline Setup**
```bash
# Navigate to RAG pipeline directory
cd rag_pipeline

# Install RAG pipeline dependencies
pip install -r requirements.txt

# Ingest training data for RAG context
python cli.py --ingest "../data/training-data-RAG.csv"

# Verify RAG pipeline is working
python cli.py --info
```

### **OpenAI API Key**
Set your OpenAI API key in one of these ways:
1. Environment variable: `export OPENAI_API_KEY="your-key"`
2. Config file: Edit `rag_pipeline/config.yml`
3. Interactive prompt: Enter when script runs

## 🚨 **Important Notes**

### **Rate Limiting**
- The script includes a 0.1-second delay between requests
- For large datasets, consider using the `--max-scenarios` flag
- Monitor your OpenAI API usage and costs
- **Note**: Each scenario runs 6 times (3 characters × 2 approaches), so total API calls = scenarios × 6

### **RAG Pipeline Requirements**
- Must have training data ingested in ChromaDB
- RAG pipeline must be properly configured
- Similar scenarios will be retrieved for context

### **Data Privacy**
- All scenario data is sent to OpenAI's API
- RAG context includes historical training data
- Results are saved locally and not shared

## 🔬 **Research Applications**

This comparison experiment can be used to:
- **Evaluate RAG Effectiveness**: Test if RAG improves decision quality
- **Study Context Influence**: Understand how additional context affects choices
- **Compare Decision Consistency**: Analyze if RAG makes decisions more consistent
- **Assess Character Adherence**: Test if RAG helps maintain character profiles
- **Optimize Prompt Design**: Find optimal balance of context vs. simplicity
- **Validate RAG Implementation**: Ensure RAG pipeline provides useful context

## 📝 **Example Usage Workflow**

1. **Setup RAG Pipeline**: Ensure training data is ingested
2. **Test Small Scale**: Run `test_rag_comparison.py` to verify setup
3. **Run Comparison**: Execute full experiment with chosen parameters
4. **Analyze Results**: Compare CSV files for differences
5. **Generate Insights**: Identify patterns in RAG vs non-RAG decisions
6. **Iterate**: Adjust RAG parameters or prompt design based on results

## 📊 **Sample Output Structure**

### **Non-RAG Results CSV:**
```
probe_id,character,llm_output,extracted_choice,success,use_rag
probe_10005_...,Alex,1600,1600,True,False
probe_10005_...,Brie,600,600,True,False
probe_10005_...,Chad,0,0,True,False
```

### **RAG Results CSV:**
```
probe_id,character,llm_output,extracted_choice,success,use_rag
probe_10005_...,Alex,1600,1600,True,True
probe_10005_...,Brie,600,600,True,True
probe_10005_...,Chad,0,0,True,True
```

### **Summary JSON:**
```json
{
  "total_scenarios": 1002,
  "total_runs": 6012,
  "non_rag_successful_choices": 2850,
  "rag_successful_choices": 2900,
  "non_rag_success_rate": 0.948,
  "rag_success_rate": 0.965,
  "model": "gpt-4",
  "temperature": 0.7,
  "rag_available": true
}
```

## 🔍 **How the RAG System Works**

### **Step-by-Step RAG Process**

#### **1. Setup**
```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-key"

# Install dependencies
cd rag_pipeline
pip install -r requirements.txt
```

#### **2. Build the Vector Store (ChromaDB)**
```bash
# Ingest training data into ChromaDB
python cli.py --ingest "../data/training-data-RAG.csv"

# Verify ingestion
python cli.py --info
```

**What happens during ingestion:**
- Each CSV row from `training-data-RAG.csv` is converted to readable "scenario text"
- Text is embedded using OpenAI's `text-embedding-3-small` model
- Embeddings + metadata are stored in persistent ChromaDB collection

#### **3. Run the Comparison Experiment**
```bash
# Basic experiment
python llm_rag_comparison_experiment.py --input data/Test Probes.csv --output-dir ./results

# Improved version with resume/backoff/chunking
python llm_rag_comparison_experiment_improved.py --input data/Test Probes.csv --output-dir ./results --chunk-size 500 --rate-limit-delay 2
```

#### **4. Internal RAG Process (Per Scenario)**

**Step 4a: Format the Scenario**
- From `data/Test Probes.csv`, build structured description:
  - Insurance Type, Network Status, Expense Type
  - Family Composition (children counts by age)
  - Employment/lifestyle (employee_type, distance, travel, housing)
  - Health factors (visits, chronic conditions %, sports %)

**Step 4b: Retrieve Similar Scenarios**
- Embed the scenario text
- Query ChromaDB for top-k nearest neighbors (default k=3)
- Format results as:
  ```
  Similar Scenario 1:
  Insurance Type: DEDUCTIBLE | Network Status: IN-NETWORK | Family Composition: 2 total children...
  
  Similar Scenario 2:
  Insurance Type: DEDUCTIBLE | Network Status: OUT-OF-NETWORK | Family Composition: 1 total children...
  
  Similar Scenario 3:
  Insurance Type: DEDUCTIBLE | Network Status: TIER 1 NETWORK | Family Composition: 3 total children...
  ```

**Step 4c: Build Final Prompt**
- Include persona-specific risk-aversion guidance (Alex/Brie/Chad)
- Append "Additional Context from Similar Insurance Scenarios" block
- List options A–D with their values (`val1..val4`)
- Instruct model to respond with chosen value (e.g., 1600, 600, 200, 0)

**Step 4d: Call OpenAI and Extract Choice**
- Send prompt to chat completion API
- Extract choice from response:
  - If A–D or "Option B" → map to letter
  - Else extract first number token (e.g., 600)

**Step 4e: Record Results**
- Save with-RAG results to `results/rag_results.csv`
- Save without-RAG results to `results/non_rag_results.csv`
- Generate summary JSON with success rates

### **Concrete Example**

**Sample Scenario from Test Probes:**
- `probe=DEDUCTIBLE`, `network_status=IN-NETWORK`, `expense_type=COST IN $`
- Children: 1 under 4, 0 under 12/18/26
- Employment: `Salaried`, 12 miles, Travel Known=Yes, Owns
- Health: 3 visits, 10% chronic, 25% sports
- Options: `val1=1600`, `val2=600`, `val3=200`, `val4=0`

**RAG Retrieval Result:**
- Similar scenarios show families with IN-NETWORK deductible choices leaning toward 600 or 200

**Final Prompt to Brie (risk aversion 0.5):**
```
[Persona guidance for Brie...]

Scenario: Insurance Type: DEDUCTIBLE
Network Status: IN-NETWORK
Expense Type: COST IN $
Family Composition: 1 total children
  - Under 4: 1
  - Under 12: 0
  - Under 18: 0
  - Under 26: 0
Employment Type: Salaried
Distance to Employer HQ: 12 miles
Travel Location Known: Yes
Housing Status: Owns
Medical Visits (Previous Year): 3
Family Members with Chronic Conditions: 10%
Family Members who Play Sports: 25%

Additional Context from Similar Insurance Scenarios:
Similar Scenario 1:
Insurance Type: DEDUCTIBLE | Network Status: IN-NETWORK | Family Composition: 2 total children...

Similar Scenario 2:
Insurance Type: DEDUCTIBLE | Network Status: IN-NETWORK | Family Composition: 1 total children...

Similar Scenario 3:
Insurance Type: DEDUCTIBLE | Network Status: TIER 1 NETWORK | Family Composition: 3 total children...

Your Task
For the given probe, your risk aversion level is 0.5 and you have following available options:
A. 1600
B. 600
C. 200
D. 0

As Brie with risk aversion = 0.5, choose the option that best aligns with your decision-maker's profile and risk aversion level.

Respond with just the value of your chosen option (e.g., 1600, 600, 200, or 0).
```

**Model Response:** "600"
**Extracted Choice:** 600
**Result:** Saved with `use_rag=True`, `extracted_choice=600`, `success=True`

### **Optimized Pipeline (for Large Datasets)**

For processing 69,000+ rows efficiently:
```bash
# Use optimized ingester
python cli_optimized.py --ingest "../data/training-data-RAG.csv"

# Features:
# - Batch embeddings (1000+ docs per API call)
# - Parallel processing (10+ workers)
# - Memory optimization
# - Progress tracking
```

## 🔍 **Interpreting Results**

### **Success Rate Comparison**
- Higher RAG success rate suggests better response parsing
- Lower RAG success rate might indicate context confusion

### **Choice Pattern Analysis**
- **Consistent Choices**: RAG and non-RAG produce same decisions
- **Divergent Choices**: RAG influences decision-making
- **Character-Specific Patterns**: Different characters respond differently to RAG

### **Quality Assessment**
- **RAG Improves**: Better decisions with additional context
- **RAG Confuses**: Additional context makes decisions worse
- **No Impact**: RAG doesn't significantly affect decisions

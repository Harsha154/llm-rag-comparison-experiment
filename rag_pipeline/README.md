# RAG Pipeline for Insurance Dataset

A Retrieval-Augmented Generation (RAG) pipeline for insurance scenario analysis using ChromaDB and OpenAI.

## 🚀 **NEW: Optimized Version for Large Datasets**

This repository now includes an **optimized version** that can process **69,000+ rows in minutes** using:
- **Batch Processing**: Process 1000+ documents per API call
- **Parallelization**: 10+ concurrent workers
- **Memory Optimization**: Efficient chunking and processing
- **Progress Tracking**: Real-time progress bars and logging

### **Performance Comparison:**
- **Original**: ~1-2 documents/second (sequential processing)
- **Optimized**: ~50-100+ documents/second (batch + parallel)
- **69,000 rows**: Estimated 10-20 minutes vs 10+ hours

## 🚀 Features

- **CSV Data Ingestion**: Load insurance scenarios from CSV files into ChromaDB
- **Vector Search**: Find similar insurance scenarios using semantic search
- **RAG Chat**: Interactive chat system with context-aware responses
- **OpenAI Integration**: Uses OpenAI embeddings and GPT-4 for intelligent responses
- **Persistent Storage**: ChromaDB for efficient vector storage and retrieval

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for OpenAI API calls

## 🛠️ Installation

1. **Navigate to the RAG pipeline directory:**
   ```bash
   cd rag_pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key:**
   
   **Option A: Environment Variable**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   **Option B: Configuration File**
   Edit `config.yml` and replace `"your-openai-api-key-here"` with your actual API key.

## 📁 Project Structure

```
rag_pipeline/
├── rag_pipeline.py              # Original RAG pipeline (sequential)
├── rag_pipeline_optimized.py    # Optimized RAG pipeline (batch + parallel)
├── cli.py                       # Original command-line interface
├── cli_optimized.py             # Optimized command-line interface
├── performance_comparison.py    # Performance comparison script
├── example_usage.py             # Example usage scripts
├── test_rag.py                  # Test script
├── config.yml                   # Configuration file
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🎯 Usage

### **For Large Datasets (69,000+ rows) - Use Optimized Version**

1. **Ingest CSV data with optimized processing:**
   ```bash
   python cli_optimized.py --ingest "path/to/your/insurance_data.csv"
   ```

2. **Test performance with sample data:**
   ```bash
   python cli_optimized.py --test-performance
   ```

3. **Customize batch size and workers:**
   ```bash
   python cli_optimized.py --ingest "data.csv" --batch-size 2000 --workers 15
   ```

4. **Start interactive chat:**
   ```bash
   python cli_optimized.py --chat
   ```

### **For Small Datasets - Use Original Version**

1. **Ingest CSV data:**
   ```bash
   python cli.py --ingest "path/to/your/insurance_data.csv"
   ```

2. **Start interactive chat:**
   ```bash
   python cli.py --chat
   ```

3. **Run a single query:**
   ```bash
   python cli.py --query "What insurance option is best for a young family?"
   ```

4. **View collection information:**
   ```bash
   python cli.py --info
   ```

5. **Delete collection:**
   ```bash
   python cli.py --delete
   ```

### **Performance Comparison**

Run the performance comparison to see the difference:
```bash
python performance_comparison.py
```

### Programmatic Usage

#### **Optimized Version (Recommended for Large Datasets)**

```python
from rag_pipeline_optimized import OptimizedRAGPipeline

# Initialize the optimized pipeline
rag = OptimizedRAGPipeline(
    collection_name="my_insurance_data",
    persist_directory="./my_chroma_db"
)

# Customize batch processing
rag.batch_size = 2000  # Process 2000 documents per batch
rag.max_workers = 15   # Use 15 parallel workers

# Ingest CSV data with optimized processing
rag.ingest_csv_optimized("path/to/insurance_data.csv")

# Query similar scenarios
results = rag.query_vector_db("young family with children", n_results=5)

# Chat with RAG
response = rag.chat_with_rag("What insurance would you recommend for me?")

# Get collection info
info = rag.get_collection_info()
```

#### **Original Version (For Small Datasets)**

```python
from rag_pipeline import RAGPipeline

# Initialize the pipeline
rag = RAGPipeline(
    collection_name="my_insurance_data",
    persist_directory="./my_chroma_db"
)

# Ingest CSV data
rag.ingest_csv("path/to/insurance_data.csv")

# Query similar scenarios
results = rag.query_vector_db("young family with children", n_results=5)

# Chat with RAG
response = rag.chat_with_rag("What insurance would you recommend for me?")

# Get collection info
info = rag.get_collection_info()
```

### Example Usage Script

Run the comprehensive example:
```bash
python example_usage.py
```

## 🔧 Configuration

Edit `config.yml` to customize:

- **OpenAI Models**: Choose different models for embeddings and chat
- **ChromaDB Settings**: Configure persistence directory and collection names
- **RAG Parameters**: Adjust context count, temperature, and token limits
- **Processing Options**: Set batch sizes and progress intervals

## 📊 Data Format

The pipeline expects CSV files with the following columns:
- Probe ID
- Age
- Gender
- Marital Status
- Children
- Income
- Health Status
- Occupation
- Location
- Option 1 (Cost)
- Option 2 (Cost)
- Option 3 (Cost)
- Option 4 (Cost)
- Ground Truth

## 💡 Example Queries

- "What's the best insurance for a single person?"
- "How does age affect insurance recommendations?"
- "What options are available for families with children?"
- "Which insurance is most cost-effective for young professionals?"
- "What should I consider when choosing between different insurance options?"

## 🔍 Advanced Features

### Custom Embeddings
The pipeline uses OpenAI's `text-embedding-3-small` model for creating embeddings. You can modify the model in the configuration.

### Context Management
Adjust the number of similar scenarios used as context by changing the `n_context` parameter in `chat_with_rag()`.

### Batch Processing
For large datasets, the pipeline processes documents in batches to manage memory usage and API rate limits.

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your API key is set correctly in environment variables or config file
   - Check that your OpenAI account has sufficient credits

2. **CSV File Not Found**
   - Verify the file path is correct
   - Ensure the file has the expected column structure

3. **ChromaDB Connection Issues**
   - Check that the persist directory is writable
   - Ensure sufficient disk space

4. **Memory Issues with Large Datasets**
   - Reduce batch size in configuration
   - Process data in smaller chunks

### Logging

The pipeline provides detailed logging. Check the console output for:
- Ingestion progress
- Embedding creation status
- Query results
- Error messages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the LLM Alignable Decision-Makers research project.

## 🔗 Related Projects

- Main LLM Alignment Project: `../`
- Insurance Dataset: `../data/`
- Evaluation Scripts: `../scripts/`

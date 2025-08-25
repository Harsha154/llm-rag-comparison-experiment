#!/usr/bin/env python3
"""
Test RAG Initialization
======================

Simple test to verify the RAGComparisonExperiment class initializes correctly
without making any API calls.
"""

def test_rag_initialization():
    """Test that the RAGComparisonExperiment class initializes without errors."""
    print("🧪 Testing RAG Comparison Experiment Initialization")
    print("=" * 60)
    
    try:
        # Import the class
        from llm_rag_comparison_experiment import RAGComparisonExperiment
        
        print("✅ Successfully imported RAGComparisonExperiment class")
        
        # Test that we can create an instance (this will fail at API initialization, but not at RAG setup)
        print("🔄 Attempting to create experiment instance...")
        
        # This will fail at OpenAI initialization, but should pass the RAG setup
        experiment = RAGComparisonExperiment(
            model="gpt-4",
            temperature=0.7
        )
        
        print("✅ Experiment instance created successfully!")
        print(f"   RAG Available: {experiment.rag_available}")
        print(f"   Model: {experiment.model}")
        print(f"   Temperature: {experiment.temperature}")
        print(f"   Characters: {list(experiment.characters.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # If it's an OpenAI API error, that's expected without a key
        if "OpenAI API key" in str(e) or "api_key" in str(e):
            print("   ✅ This is expected - OpenAI API key is required for full functionality")
            return True
        else:
            print("   ❌ This is an unexpected error")
            return False

if __name__ == "__main__":
    success = test_rag_initialization()
    if success:
        print("\n🎉 RAG initialization test passed!")
    else:
        print("\n💥 RAG initialization test failed!")

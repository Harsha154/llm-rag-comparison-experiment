#!/usr/bin/env python3
"""
RAG Pipeline Core Module
========================

This module provides the core RAG (Retrieval-Augmented Generation) functionality
for insurance dataset analysis using ChromaDB and OpenAI.
"""

import os
import logging
import pandas as pd
import chromadb
from openai import OpenAI
from typing import List, Dict, Any, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG Pipeline for insurance dataset using ChromaDB and OpenAI."""
    
    def __init__(self, collection_name: str = "insurance_scenarios", persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG pipeline.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize OpenAI client
        self.openai_client = self._initialize_openai()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")
    
    def _initialize_openai(self) -> OpenAI:
        """Initialize OpenAI client with credentials."""
        # Try to get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            # Try to get from config file
            config_path = "config.yml"
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
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise
    
    def _create_scenario_text(self, row: List[str], headers: List[str]) -> str:
        """Create a text description of an insurance scenario from CSV row."""
        scenario_parts = []
        
        # Map column names to more readable descriptions
        column_mapping = {
            'Probe ID': 'Scenario ID',
            'Age': 'Age',
            'Gender': 'Gender',
            'Marital Status': 'Marital Status',
            'Children': 'Number of Children',
            'Income': 'Annual Income',
            'Health Status': 'Health Status',
            'Occupation': 'Occupation',
            'Location': 'Location',
            'Option 1': 'Option 1 Cost',
            'Option 2': 'Option 2 Cost', 
            'Option 3': 'Option 3 Cost',
            'Option 4': 'Option 4 Cost',
            'Ground Truth': 'Recommended Option'
        }
        
        for i, header in enumerate(headers):
            if i < len(row):
                value = row[i]
                readable_header = column_mapping.get(header, header)
                scenario_parts.append(f"{readable_header}: {value}")
        
        return " | ".join(scenario_parts)
    
    def ingest_csv(self, csv_file_path: str) -> None:
        """
        Ingest CSV data into ChromaDB.
        
        Args:
            csv_file_path: Path to the CSV file
        """
        logger.info(f"Starting CSV ingestion from: {csv_file_path}")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            # Convert DataFrame to list of lists for processing
            data = df.values.tolist()
            headers = df.columns.tolist()
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, row in enumerate(data):
                # Create scenario text
                scenario_text = self._create_scenario_text(row, headers)
                
                # Create metadata
                metadata = {
                    'row_index': i,
                    'probe_id': row[0] if row else f"INS_{i}",
                    'age': row[1] if len(row) > 1 else None,
                    'gender': row[2] if len(row) > 2 else None,
                    'marital_status': row[3] if len(row) > 3 else None,
                    'children': row[4] if len(row) > 4 else None,
                    'income': row[5] if len(row) > 5 else None,
                    'health_status': row[6] if len(row) > 6 else None,
                    'occupation': row[7] if len(row) > 7 else None,
                    'location': row[8] if len(row) > 8 else None,
                    'option_1_cost': row[9] if len(row) > 9 else None,
                    'option_2_cost': row[10] if len(row) > 10 else None,
                    'option_3_cost': row[11] if len(row) > 11 else None,
                    'option_4_cost': row[12] if len(row) > 12 else None,
                    'ground_truth': row[13] if len(row) > 13 else None
                }
                
                documents.append(scenario_text)
                metadatas.append(metadata)
                ids.append(f"scenario_{i}")
                
                # Log progress every 100 rows
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} scenarios...")
            
            # Create embeddings and add to ChromaDB
            logger.info("Creating embeddings...")
            embeddings = []
            
            for i, doc in enumerate(documents):
                try:
                    embedding = self._create_embedding(doc)
                    embeddings.append(embedding)
                    
                    # Log progress every 50 embeddings
                    if (i + 1) % 50 == 0:
                        logger.info(f"Created {i + 1} embeddings...")
                        
                except Exception as e:
                    logger.error(f"Error creating embedding for document {i}: {e}")
                    # Use zero embedding as fallback
                    embeddings.append([0.0] * 1536)  # text-embedding-3-small dimension
            
            # Add to ChromaDB collection
            logger.info("Adding documents to ChromaDB...")
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully ingested {len(documents)} scenarios into ChromaDB")
            
        except Exception as e:
            logger.error(f"Error ingesting CSV: {e}")
            raise
    
    def query_vector_db(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database for similar scenarios.
        
        Args:
            query: The query text
            n_results: Number of results to return
            
        Returns:
            List of similar scenarios with metadata
        """
        try:
            # Create embedding for query
            query_embedding = self._create_embedding(query)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying vector database: {e}")
            raise
    
    def chat_with_rag(self, user_message: str, n_context: int = 3) -> str:
        """
        Chat with the RAG system using retrieved context.
        
        Args:
            user_message: User's message
            n_context: Number of similar scenarios to include as context
            
        Returns:
            AI response with RAG context
        """
        try:
            # Query vector database for relevant scenarios
            similar_scenarios = self.query_vector_db(user_message, n_results=n_context)
            
            # Build context from similar scenarios
            context_parts = []
            for i, scenario in enumerate(similar_scenarios, 1):
                context_parts.append(f"Similar Scenario {i}:\n{scenario['document']}\n")
            
            context = "\n".join(context_parts)
            
            # Create system prompt
            system_prompt = f"""You are an insurance advisor assistant. You have access to similar insurance scenarios to help provide informed advice.

Available context from similar scenarios:
{context}

Please provide helpful advice based on the user's question and the similar scenarios above. Be specific and reference the context when relevant."""

            # Get AI response
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in RAG chat: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def delete_collection(self) -> None:
        """Delete the current collection."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

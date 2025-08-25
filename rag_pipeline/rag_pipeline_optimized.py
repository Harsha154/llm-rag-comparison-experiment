#!/usr/bin/env python3
"""
Optimized RAG Pipeline with Batch Processing and Parallelization
===============================================================

This module provides an optimized RAG pipeline that can process 69,000+ rows
in minutes using batch processing and parallelization.
"""

import os
import logging
import pandas as pd
import chromadb
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple
import yaml
from dotenv import load_dotenv
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import json

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedRAGPipeline:
    """Optimized RAG Pipeline with batch processing and parallelization."""
    
    def __init__(self, collection_name: str = "insurance_scenarios", persist_directory: str = "./chroma_db"):
        """
        Initialize the optimized RAG pipeline.
        
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
        
        # Configuration for batch processing
        self.batch_size = 1000  # Process 1000 documents per batch
        self.max_workers = 10   # Number of parallel workers
        self.max_retries = 3    # Number of retries for failed requests
        
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
    
    def _create_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of texts using OpenAI API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 1536 for _ in texts]
    
    def _process_batch(self, batch_data: Tuple[List[str], List[Dict], List[str], int]) -> Tuple[List[List[float]], List[Dict], List[str]]:
        """
        Process a batch of documents.
        
        Args:
            batch_data: Tuple of (documents, metadatas, ids, batch_index)
            
        Returns:
            Tuple of (embeddings, metadatas, ids)
        """
        documents, metadatas, ids, batch_index = batch_data
        
        logger.info(f"Processing batch {batch_index + 1} with {len(documents)} documents")
        
        # Create embeddings for the batch
        embeddings = self._create_batch_embeddings(documents)
        
        logger.info(f"Completed batch {batch_index + 1}")
        return embeddings, metadatas, ids
    
    def _chunk_data(self, data: List, chunk_size: int) -> List[List]:
        """Split data into chunks of specified size."""
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    def ingest_csv_optimized(self, csv_file_path: str) -> None:
        """
        Optimized CSV ingestion with batch processing and parallelization.
        
        Args:
            csv_file_path: Path to the CSV file
        """
        start_time = time.time()
        logger.info(f"Starting optimized CSV ingestion from: {csv_file_path}")
        
        try:
            # Read CSV file
            logger.info("Reading CSV file...")
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            # Convert DataFrame to list of lists for processing
            data = df.values.tolist()
            headers = df.columns.tolist()
            
            # Prepare data for ChromaDB
            logger.info("Preparing documents and metadata...")
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
            
            logger.info(f"Prepared {len(documents)} documents for processing")
            
            # Split data into batches
            doc_batches = self._chunk_data(documents, self.batch_size)
            metadata_batches = self._chunk_data(metadatas, self.batch_size)
            id_batches = self._chunk_data(ids, self.batch_size)
            
            logger.info(f"Split data into {len(doc_batches)} batches of size {self.batch_size}")
            
            # Process batches in parallel
            all_embeddings = []
            all_metadatas = []
            all_ids = []
            
            # Create batch data tuples
            batch_data = [
                (doc_batches[i], metadata_batches[i], id_batches[i], i)
                for i in range(len(doc_batches))
            ]
            
            logger.info("Starting parallel batch processing...")
            
            # Process batches with ThreadPoolExecutor for parallelization
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all batch processing tasks
                future_to_batch = {
                    executor.submit(self._process_batch, batch): batch[3] 
                    for batch in batch_data
                }
                
                # Collect results as they complete
                for future in tqdm(as_completed(future_to_batch), total=len(batch_data), desc="Processing batches"):
                    batch_index = future_to_batch[future]
                    try:
                        embeddings, metadatas, ids = future.result()
                        all_embeddings.extend(embeddings)
                        all_metadatas.extend(metadatas)
                        all_ids.extend(ids)
                        logger.info(f"Completed batch {batch_index + 1}")
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_index + 1}: {e}")
            
            # Add all data to ChromaDB in chunks to avoid memory issues
            logger.info("Adding data to ChromaDB...")
            chunk_size = 5000  # Add to ChromaDB in smaller chunks
            
            for i in range(0, len(all_embeddings), chunk_size):
                end_idx = min(i + chunk_size, len(all_embeddings))
                chunk_embeddings = all_embeddings[i:end_idx]
                chunk_metadatas = all_metadatas[i:end_idx]
                chunk_ids = all_ids[i:end_idx]
                
                self.collection.add(
                    documents=[meta.get('probe_id', '') for meta in chunk_metadatas],  # Use probe_id as document text
                    embeddings=chunk_embeddings,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )
                logger.info(f"Added chunk {i//chunk_size + 1} to ChromaDB ({end_idx}/{len(all_embeddings)} documents)")
            
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"Successfully ingested {len(all_embeddings)} scenarios into ChromaDB in {processing_time:.2f} seconds")
            logger.info(f"Average processing speed: {len(all_embeddings)/processing_time:.2f} documents/second")
            
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
            query_embedding = self._create_batch_embeddings([query])[0]
            
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

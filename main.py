import logging
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os
from pathlib import Path
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Union, Any  # Add typing imports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MilvusVectorDB:
    def __init__(self):
        self.dimension = 1024
        self.collection_name = "markdown_collection"
        self.model_name = "BAAI/bge-large-en"
        
        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")
        
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        logger.info(f"Initialized MilvusVectorDB with model: {self.model_name}")
        
        self.processed_files = self.load_processed_files()

    def load_processed_files(self) -> List[str]:
        """Load list of processed files"""
        try:
            with open('processed_files.json', 'r') as f:
                data = json.load(f)
                return data.get('files', [])
        except FileNotFoundError:
            return []

    def update_processed_files(self, new_file: str):
        """Add newly processed file to tracking"""
        if new_file not in self.processed_files:
            self.processed_files.append(new_file)
            with open('processed_files.json', 'w') as f:
                json.dump({
                    'files': self.processed_files,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)

    def get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file content"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def check_unprocessed_files(self, directory_path: str) -> List[Path]:
        """Check which files in the directory haven't been processed"""
        all_files = []
        for ext in ['*.md', '*.markdown']:
            all_files.extend(list(Path(directory_path).rglob(ext)))
        
        unprocessed_files = []
        for file_path in all_files:
            file_path_str = str(file_path)
            file_hash = self.get_file_hash(file_path)
            
            # Check if file is new or has been modified
            if (file_path_str not in self.processed_files or 
                self.processed_files[file_path_str]['hash'] != file_hash):
                unprocessed_files.append(file_path)
        
        return unprocessed_files

    def create_collection(self):
        """Create collection if it doesn't exist, otherwise use existing one"""
        if self.collection_name not in utility.list_collections():
            logger.info("Creating new collection with schema...")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
            schema = CollectionSchema(fields=fields, description="Markdown embeddings collection")
            collection = Collection(name=self.collection_name, schema=schema)
            
            # Create index for vector field
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index("embedding", index_params)
            logger.info("Collection created and indexed successfully")
        else:
            logger.info("Using existing collection")
            collection = Collection(self.collection_name)
            
        return collection

    def get_embedding(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embedding = model_output.last_hidden_state[:, 0].numpy()
        return embedding[0]

    def read_markdown_file(self, file_path):
        """Read content from markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_markdown_files(self, directory):
        """Get all markdown files from directory."""
        markdown_files = []
        for ext in ['*.md', '*.markdown']:
            markdown_files.extend(Path(directory).glob(ext))
        return markdown_files

    def chunk_text(self, text, max_length=65000):
        """Split text into chunks of maximum length."""
        logger.debug(f"Chunking text of length {len(text)}")
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= max_length:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())

        # If any chunk is still too long, split it further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_length:
                # Split into smaller pieces
                while chunk:
                    final_chunks.append(chunk[:max_length])
                    chunk = chunk[max_length:]
            else:
                final_chunks.append(chunk)

        logger.debug(f"Text split into {len(final_chunks)} chunks")
        return final_chunks

    def process_directory(self, directory_path: str) -> None:
        """Process markdown files that haven't been processed yet"""
        logger.info(f"Scanning directory: {directory_path}")
        
        # Get all markdown files
        markdown_files = []
        for ext in ['*.md', '*.markdown']:
            markdown_files.extend(list(Path(directory_path).rglob(ext)))
        
        # Filter out already processed files
        files_to_process = [
            f for f in markdown_files 
            if str(f) not in self.processed_files
        ]
        
        if not files_to_process:
            logger.info("No new files to process")
            return
        
        logger.info(f"Found {len(files_to_process)} new files to process")
        collection = Collection(self.collection_name)
        
        for i, file_path in enumerate(files_to_process, 1):
            try:
                logger.info(f"Processing file {i}/{len(files_to_process)}: {file_path}")
                content = self.read_markdown_file(file_path)
                chunks = self.chunk_text(content)
                
                for chunk in chunks:
                    embedding = self.get_embedding(chunk)
                    entities = [{
                        "text": chunk,
                        "embedding": embedding.tolist()
                    }]
                    collection.insert(entities)
                
                self.update_processed_files(str(file_path))
                logger.info(f"Processed and tracked file: {file_path}")
                
                # Flush every 10 files or on last file
                if i % 10 == 0 or i == len(files_to_process):
                    collection.flush()
                    logger.info("Flushed to database")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

    def search(self, query_text, top_k=5):
        logger.info(f"Searching for: '{query_text[:50]}...' with top_k={top_k}")
        collection = Collection(self.collection_name)
        collection.load()
        
        logger.debug("Generating query embedding")
        query_embedding = self.get_embedding(query_text)
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        logger.debug("Executing search")
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        
        logger.info(f"Found {len(results[0])} matches")
        return [(hit.entity.get('text'), hit.distance) for hit in results[0]]

# Modify the main block for simpler usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process markdown files into vector DB')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing markdown files')
    parser.add_argument('--list', action='store_true', help='List processed files')
    args = parser.parse_args()

    db = MilvusVectorDB()
    
    if args.list:
        print("\nProcessed files:")
        for file in db.processed_files:
            print(f"- {file}")
    else:
        db.create_collection()
        db.process_directory(args.dir)
    
    # Search example
    search_query = "your search query"
    logger.info(f"Performing search with query: {search_query}")
    results = db.search(search_query, top_k=2)
    
    logger.info("Search results:")
    for i, (text, score) in enumerate(results, 1):
        logger.info(f"\nResult {i}:")
        logger.info(f"Similarity Score: {score}")
        logger.info(f"Content Preview: {text[:200]}...")



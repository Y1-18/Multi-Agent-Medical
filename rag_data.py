
import sys
import json
import logging
import os
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

# CRITICAL: Set PyTorch memory allocation strategy BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to path if needed
sys.path.append(str(Path(__file__).parent.parent))

# Import your components
from agents.rag_agent import MedicalRAG
from config import Config

import argparse

# Initialize parser
parser = argparse.ArgumentParser(description="Ingest documents into the Medical RAG system.")

# Add arguments
parser.add_argument("--file", type=str, required=False, help="Enter file path to ingest")
parser.add_argument("--dir", type=str, required=False, help="Enter directory path of files to ingest")

# Parse arguments
args = parser.parse_args()

# Load configuration
config = Config()

rag = MedicalRAG(config)

def data_ingestion():
    """Ingest documents into the RAG system."""
    
    if args.file:
        # Define path to file
        file_path = args.file
        print(f"\n{'='*60}")
        print(f"Ingesting single file: {file_path}")
        print(f"{'='*60}\n")
        
        # Process and ingest the file
        result = rag.ingest_file(file_path)
        
    elif args.dir:
        # Define path to dir
        dir_path = args.dir
        print(f"\n{'='*60}")
        print(f"Ingesting directory: {dir_path}")
        print(f"{'='*60}\n")
        
        # Process and ingest the files
        result = rag.ingest_directory(dir_path)
    else:
        print("\nError: Please provide either --file or --dir argument")
        parser.print_help()
        return False

    # Print results
    print(f"\n{'='*60}")
    print("INGESTION RESULTS")
    print(f"{'='*60}")
    print(json.dumps(result, indent=2))
    print(f"{'='*60}\n")
    
    # Print summary
    if result.get("success"):
        print(f"✓ Successfully ingested {result.get('documents_ingested', 0)} document(s)")
        print(f"✓ Total chunks processed: {result.get('chunks_processed', 0)}")
        
        if result.get('failed_documents', 0) > 0:
            print(f"\n⚠ {result['failed_documents']} document(s) failed:")
            for failed_file in result.get('failed_files', []):
                print(f"  - {failed_file['file']}")
                print(f"    Error: {failed_file['error'][:100]}...")
    else:
        print(f"✗ Ingestion failed: {result.get('error', 'Unknown error')}")
    
    print(f"\nProcessing time: {result.get('processing_time', 0):.2f} seconds")

    return result.get("success", False)

# Run ingestion
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MEDICAL RAG SYSTEM - DOCUMENT INGESTION")
    print("="*60)
    
    try:
        # Clear GPU cache before starting (if available)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"✓ Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                print(f"✓ Initial GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        except ImportError:
            print("ℹ PyTorch not available or no GPU detected")
        
        print()
        
        ingestion_success = data_ingestion()
        
        if ingestion_success:
            print("\n✓ Document ingestion completed successfully!")
        else:
            print("\n✗ Document ingestion failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠ Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Fatal error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
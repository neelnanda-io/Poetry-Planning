import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from tqdm import tqdm
import gc

# Configure cache directory
CACHE_DIR = "/workspace/cache"
os.environ['HF_HOME'] = CACHE_DIR

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(CACHE_DIR, 'download.log'))
    ]
)

# Define the models to download
MODELS = [
    "google/gemma-2-2b", 
    "google/gemma-2-9b", 
    "google/gemma-2-27b", 
    "google/gemma-3-4b", 
    "google/gemma-3-12b", 
    "google/gemma-3-27b", 
]

def download_model(model_name):
    """
    Download and cache a model and its tokenizer
    """
    try:
        logging.info(f"Starting download of {model_name}")
        
        # Download and cache the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        
        # Determine device based on availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device} for {model_name}")
        
        # Download and cache the model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            device_map="auto"  # Automatically handle model placement
        )
        
        # Clear model and tokenizer from memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        logging.info(f"Successfully downloaded and cached {model_name}")
        return True
    except Exception as e:
        logging.error(f"Error downloading {model_name}: {str(e)}", exc_info=True)
        return False

def main():
    successful = 0
    failed = 0
    failed_models = []
    
    logging.info(f"Starting download process with cache directory: {CACHE_DIR}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create a progress bar
    with tqdm(total=len(MODELS), desc="Downloading models") as pbar:
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all download tasks
            future_to_model = {
                executor.submit(download_model, model): model 
                for model in MODELS
            }
            
            # Process completed downloads
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    if future.result():
                        successful += 1
                    else:
                        failed += 1
                        failed_models.append(model)
                except Exception as e:
                    logging.error(f"Download failed for {model}: {str(e)}", exc_info=True)
                    failed += 1
                    failed_models.append(model)
                pbar.update(1)
    
    # Print summary
    logging.info(f"\nDownload Summary:")
    logging.info(f"Successfully downloaded: {successful} models")
    logging.info(f"Failed downloads: {failed} models")
    if failed_models:
        logging.info(f"Failed models: {', '.join(failed_models)}")
    logging.info(f"Detailed logs available at: {os.path.join(CACHE_DIR, 'download.log')}")

if __name__ == "__main__":
    main() 

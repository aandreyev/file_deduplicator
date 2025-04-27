import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
load_dotenv()
logging.info("Loaded environment variables.")

# --- Configuration ---
# Match these with your deduplicator.py and Supabase setup
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") # Use Service Role Key for updates
TABLE_NAME = "documents" 
TEXT_COLUMN = "extracted_content"
ID_COLUMN = "id"
EMBEDDING_COLUMN = "embedding" # The new vector column
MODEL_NAME = 'all-MiniLM-L6-v2' # Make sure this matches deduplicator.py
BATCH_SIZE = 50 # Process documents in batches

logging.info(f"Configuration: TABLE_NAME='{TABLE_NAME}', TEXT_COLUMN='{TEXT_COLUMN}', ID_COLUMN='{ID_COLUMN}', EMBEDDING_COLUMN='{EMBEDDING_COLUMN}'")

# --- Input Validation ---
if not all([SUPABASE_URL, SUPABASE_KEY, TABLE_NAME, TEXT_COLUMN, ID_COLUMN, EMBEDDING_COLUMN, MODEL_NAME]):
    logging.error("One or more configuration variables are missing. Please check your .env file and script configuration.")
    sys.exit(1)

# --- Initialize Supabase Client ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logging.info("Supabase client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Supabase client: {e}")
    sys.exit(1)

# --- Initialize Sentence Transformer Model ---
try:
    logging.info(f"Loading sentence transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    logging.info("Sentence transformer model loaded successfully.")
    # Get embedding dimension dynamically from the loaded model
    EMBEDDING_DIM = model.get_sentence_embedding_dimension()
    logging.info(f"Model embedding dimension: {EMBEDDING_DIM}") 
except Exception as e:
    logging.error(f"Failed to load sentence transformer model '{MODEL_NAME}': {e}")
    sys.exit(1)

# --- Main Backfill Logic ---
def backfill():
    logging.info("Starting embedding backfill process...")
    
    total_processed = 0
    offset = 0
    
    while True:
        logging.info(f"Fetching documents from offset {offset}, limit {BATCH_SIZE}...")
        try:
            # Fetch documents that don't have an embedding yet (or fetch all if you want to overwrite)
            # Fetching only ID and TEXT_COLUMN for efficiency
            response = (supabase.table(TABLE_NAME)
                        .select(f"{ID_COLUMN}, {TEXT_COLUMN}")
                        .is_(EMBEDDING_COLUMN, 'null') # Fetch only rows where embedding is NULL
                        # If you want to re-calculate all, remove the .is_ filter
                        .range(offset, offset + BATCH_SIZE - 1)
                        .execute())
            
            if not response.data:
                logging.info("No more documents found without embeddings.")
                break

            docs_batch = response.data
            logging.info(f"Fetched {len(docs_batch)} documents in this batch.")

            # Prepare texts and IDs
            ids = [doc[ID_COLUMN] for doc in docs_batch]
            texts = [doc.get(TEXT_COLUMN, "") or "" for doc in docs_batch] # Handle None or empty strings

            # Calculate embeddings
            logging.info(f"Calculating embeddings for {len(texts)} documents...")
            try:
                embeddings = model.encode(texts, show_progress_bar=True)
                # Ensure embeddings are lists of floats for Supabase pgvector
                embeddings_list = [emb.tolist() for emb in embeddings] 
                logging.info("Embeddings calculated.")
            except Exception as e:
                logging.error(f"Error calculating embeddings for batch starting at offset {offset}: {e}")
                # Decide if you want to skip the batch or stop
                offset += len(docs_batch) # Move to next batch even if this one failed
                continue 

            # Update documents in Supabase
            updates = []
            for doc_id, embedding_val in zip(ids, embeddings_list):
                 # Check embedding dimension matches expected dimension (optional but good practice)
                 if len(embedding_val) == EMBEDDING_DIM:
                    updates.append({
                        ID_COLUMN: doc_id,
                        EMBEDDING_COLUMN: embedding_val 
                    })
                 else:
                     logging.warning(f"Document ID {doc_id} generated embedding with incorrect dimension {len(embedding_val)}, expected {EMBEDDING_DIM}. Skipping update.")

            if updates:
                logging.info(f"Updating {len(updates)} documents in Supabase...")
                try:
                    # Use upsert for batch update (update based on ID_COLUMN match)
                    supabase.table(TABLE_NAME).upsert(updates).execute()
                    logging.info("Batch update successful.")
                    total_processed += len(updates)
                except Exception as e:
                    logging.error(f"Error updating batch in Supabase: {e}")
                    # Optional: Implement retry logic or log failed IDs
            else:
                 logging.info("No valid embeddings generated in this batch to update.")


            # Check if we fetched less than BATCH_SIZE, meaning it's the last page
            if len(docs_batch) < BATCH_SIZE:
                logging.info("Processed the last batch.")
                break
                
            # Move to the next batch
            offset += len(docs_batch)

        except Exception as e:
            logging.error(f"An error occurred during the fetch/process loop at offset {offset}: {e}")
            # Decide on retry logic or break
            break
            
    logging.info(f"Embedding backfill process completed. Total documents updated: {total_processed}")

if __name__ == "__main__":
    backfill() 
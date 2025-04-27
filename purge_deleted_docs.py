import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from postgrest.exceptions import APIError # To catch specific DB errors like unique constraints

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Purge soft-deleted documents and their associated storage files.")
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Simulate the process without actually deleting files or database rows."
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=50,
    help="Number of documents to process in each batch."
)
args = parser.parse_args()

if args.dry_run:
    logging.warning("---------- DRY RUN MODE ACTIVATED ----------")
    logging.warning("No files or database records will be deleted.")
    logging.warning("------------------------------------------")

# --- Load Environment Variables ---
load_dotenv()
logging.info("Loaded environment variables.")

# --- Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") # MUST be Service Role Key
DOCUMENTS_TABLE = "documents"
PROCESSED_TABLE = "processed_duplicates"
STORAGE_BUCKET = "documents" # As provided by user

# Columns to fetch from documents table
ID_COL = "id" # bigint
UNIQUE_ID_COL = "unique_id" # text, PK for processed_duplicates
STORAGE_PATH_COL = "storage_path" # text, path within the bucket
FILENAME_COL = "original_filename" # text, for logging
IS_DELETED_COL = "is_deleted" # boolean

logging.info(f"Configuration: DOCUMENTS_TABLE='{DOCUMENTS_TABLE}', PROCESSED_TABLE='{PROCESSED_TABLE}', STORAGE_BUCKET='{STORAGE_BUCKET}'")

# --- Input Validation ---
if not all([SUPABASE_URL, SUPABASE_KEY, DOCUMENTS_TABLE, PROCESSED_TABLE, STORAGE_BUCKET, ID_COL, UNIQUE_ID_COL, STORAGE_PATH_COL, FILENAME_COL, IS_DELETED_COL]):
    logging.error("One or more configuration variables are missing. Please check your .env file and script configuration.")
    sys.exit(1)

# --- Initialize Supabase Client ---
try:
    # Increase timeout for potentially long-running delete operations
    options = ClientOptions(postgrest_client_timeout=60, storage_client_timeout=60)
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=options)
    logging.info("Supabase client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Supabase client: {e}")
    sys.exit(1)

# --- Main Purge Logic ---
def purge_deleted_documents():
    logging.info("Starting purge process for soft-deleted documents...")
    total_processed_docs = 0
    total_storage_deleted = 0
    total_logged = 0
    total_rows_deleted = 0
    total_skipped_storage_error = 0
    total_skipped_storage_not_found = 0 # New counter
    total_skipped_log = 0
    total_skipped_final_delete = 0
    offset = 0
    batch_size = args.batch_size

    while True:
        logging.info(f"Fetching batch of soft-deleted documents from offset {offset}, limit {batch_size}...")
        try:
            response = (supabase.table(DOCUMENTS_TABLE)
                        .select(f"{ID_COL}, {UNIQUE_ID_COL}, {STORAGE_PATH_COL}, {FILENAME_COL}")
                        .eq(IS_DELETED_COL, True)
                        .range(offset, offset + batch_size - 1)
                        .execute())

            if not response.data:
                logging.info("No more soft-deleted documents found.")
                break

            docs_batch = response.data
            logging.info(f"Fetched {len(docs_batch)} documents in this batch.")
            total_processed_docs += len(docs_batch)

            for doc in docs_batch:
                doc_id = doc[ID_COL]
                unique_id = doc[UNIQUE_ID_COL]
                storage_path = doc[STORAGE_PATH_COL]
                filename = doc.get(FILENAME_COL, "[Unknown Filename]") # Use .get for safety

                if not unique_id:
                    logging.warning(f"Skipping document ID {doc_id} because it lacks a unique_id needed for logging.")
                    total_skipped_log += 1
                    continue

                # 1. Attempt to Delete from Storage
                storage_delete_success = False
                if storage_path:
                    logging.info(f"Attempting to delete storage file for Doc ID {doc_id} (Unique ID: {unique_id}, Path: {storage_path})...")
                    if not args.dry_run:
                        try:
                            res = supabase.storage.from_(STORAGE_BUCKET).remove([storage_path])
                            logging.info(f"Storage deletion successful for path: {storage_path}")
                            total_storage_deleted += 1
                            storage_delete_success = True
                        except APIError as storage_error:
                            # Check if the error is specifically a "not found" type
                            is_not_found_error = (
                                "Could not find the file" in str(storage_error) or
                                (hasattr(storage_error, 'status_code') and storage_error.status_code in [400, 404]) or
                                (hasattr(storage_error, 'json') and isinstance(storage_error.json(), dict) and storage_error.json().get('message') == 'The resource was not found')
                            )
                            if is_not_found_error:
                                # --- MODIFIED BEHAVIOR --- 
                                logging.error(f"HALTING processing for Doc ID {doc_id}: Storage file not found at path: {storage_path}. This may indicate an inconsistency.")
                                total_skipped_storage_not_found += 1
                                storage_delete_success = False # Ensure we don't proceed
                                # --- /MODIFIED BEHAVIOR --- 
                            else:
                                # Other storage API errors
                                logging.error(f"Error deleting storage file {storage_path} for Doc ID {doc_id}: {storage_error}")
                                total_skipped_storage_error += 1
                                storage_delete_success = False
                        except Exception as e:
                             # Unexpected errors during storage delete
                             logging.error(f"Unexpected error deleting storage file {storage_path} for Doc ID {doc_id}: {e}")
                             total_skipped_storage_error += 1
                             storage_delete_success = False
                    else:
                        # Dry run simulation
                        logging.info(f"[DRY RUN] Would attempt delete storage file: {storage_path}")
                        # Simulate as if file exists for dry run flow unless testing not-found specifically
                        storage_delete_success = True 
                else:
                    # No storage path provided
                    logging.warning(f"No storage_path found for Doc ID {doc_id}. Cannot delete from storage. Proceeding to log/delete row...")
                    storage_delete_success = True # Allow proceeding if no path existed

                # --- Steps 2 & 3 only run if storage_delete_success is True --- 

                # 2. Log to Processed Duplicates Table
                log_success = False
                if storage_delete_success:
                    logging.info(f"Attempting to log deletion for Doc Unique ID: {unique_id}...")
                    processed_data = {
                        "unique_id": unique_id, # PK
                        "filename": filename,
                        "kept_document_unique_id": storage_path,
                        "reason": "Processed by purge script",
                    }
                    if not args.dry_run:
                        try:
                            supabase.table(PROCESSED_TABLE).insert(processed_data).execute()
                            logging.info(f"Successfully logged deletion for Unique ID: {unique_id}")
                            total_logged += 1
                            log_success = True
                        except APIError as log_error:
                            # Broader check for duplicate key violation on the primary key
                            error_detail = str(log_error).lower()
                            is_duplicate_error = (
                                "duplicate key value violates unique constraint" in error_detail and
                                (
                                    'processed_duplicates_pkey' in error_detail or # Check standard constraint name
                                    f'key ({UNIQUE_ID_COL})=({unique_id})' in error_detail # Check specific key value duplication
                                )
                            )
                            if is_duplicate_error:
                                logging.warning(f"Document with Unique ID {unique_id} already logged in {PROCESSED_TABLE}. Assuming previously processed, proceeding to final delete check...")
                                log_success = True # Treat as success if already logged
                            else:
                                # Log the specific APIError details for other DB issues
                                logging.error(f"Database APIError logging deletion for Unique ID {unique_id}: Status={getattr(log_error, 'status_code', 'N/A')}, Message={log_error}")
                                total_skipped_log += 1
                                log_success = False # Ensure we don't delete row if logging failed
                        except Exception as e:
                            # Catch any other unexpected errors during insert
                            logging.error(f"Unexpected error logging deletion for Unique ID {unique_id}: {type(e).__name__} - {e}")
                            total_skipped_log += 1
                            log_success = False
                    else:
                         logging.info(f"[DRY RUN] Would log deletion for Unique ID: {unique_id} with data: {processed_data}")
                         log_success = True # Assume success for dry run flow
                else:
                    # This branch is now also reached if storage file was not found
                    logging.warning(f"Skipping logging and final deletion for Doc ID {doc_id} because storage deletion failed or was skipped.")

                # 3. Permanently Delete from Documents Table
                final_delete_success = False
                if log_success:
                    logging.info(f"Attempting to delete document row for ID: {doc_id} (Unique ID: {unique_id})...")
                    if not args.dry_run:
                        try:
                            supabase.table(DOCUMENTS_TABLE).delete().eq(ID_COL, doc_id).execute()
                            logging.info(f"Successfully deleted row for ID: {doc_id}")
                            total_rows_deleted += 1
                            final_delete_success = True
                        except Exception as e:
                            logging.error(f"Error deleting row for ID {doc_id}: {e}")
                            total_skipped_final_delete += 1
                            final_delete_success = False
                    else:
                        logging.info(f"[DRY RUN] Would delete row from {DOCUMENTS_TABLE} for ID: {doc_id}")
                        final_delete_success = True # Assume success for dry run flow
                else:
                     # This branch is now also reached if logging failed or storage deletion failed/skipped
                    logging.warning(f"Skipping final deletion for Doc ID {doc_id} because previous steps failed or were skipped.")


            # --- Batch processing ends ---

            # Check if we fetched less than BATCH_SIZE, meaning it's the last page
            if len(docs_batch) < batch_size:
                logging.info("Processed the last batch.")
                break

            # Move to the next batch offset only if we got a full batch
            offset += batch_size

        except Exception as e:
            logging.error(f"An error occurred during the main fetch/process loop at offset {offset}: {e}")
            logging.error("Stopping process due to unexpected error.")
            break # Stop processing if there's a major error

    # --- Loop ends --- 

    logging.info("---------- Purge Process Summary ----------")
    logging.info(f"Total documents fetched with is_deleted=true: {total_processed_docs}")
    logging.info(f"Storage files deleted successfully: {total_storage_deleted}")
    logging.info(f"Deletion records logged to '{PROCESSED_TABLE}' (or already existed): {total_logged}")
    logging.info(f"Rows deleted from '{DOCUMENTS_TABLE}': {total_rows_deleted}")
    logging.warning(f"Skipped processing due to storage file not found: {total_skipped_storage_not_found}")
    logging.warning(f"Skipped storage deletions due to other errors: {total_skipped_storage_error}")
    logging.warning(f"Skipped logging due to errors or missing unique_id: {total_skipped_log}")
    logging.warning(f"Skipped final row deletion due to previous errors: {total_skipped_final_delete}")
    logging.info("------------------------------------------")
    if args.dry_run:
         logging.warning("REMINDER: This was a DRY RUN. No actual changes were made.")

if __name__ == "__main__":
    purge_deleted_documents() 
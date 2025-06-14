# File Deduplicator

Removes duplicate files for a knowledge base stored in Supabase.

## Overview

This project provides a suite of tools to identify, manage, and remove duplicate documents within a Supabase database. It leverages sentence embeddings to compare document content and offers a web interface for manual review and resolution of potential duplicates.

## Features

*   **Embedding Generation:** Calculates and stores sentence embeddings (vectors) for document content using the Sentence Transformers library.
*   **Duplicate Detection:** Identifies potential duplicate documents by comparing the cosine similarity of their embeddings.
*   **Manual Review UI:** A Streamlit web application allows users to review pairs of similar documents side-by-side, compare their content, and decide which version to keep.
*   **Soft Deletion:** Allows marking documents as "deleted" without immediately removing them from the database.
*   **Embedding Backfill:** A script (`backfill_embeddings.py`) to generate embeddings for documents that were added to the database before the embedding process was in place or for documents that are missing embeddings.
*   **Permanent Purge:** A script (`purge_deleted_docs.py`) to permanently remove documents previously marked as "deleted" and their associated files from Supabase storage.
*   **Configuration:** Utilizes a `.env` file for managing sensitive credentials and Supabase connection details.
*   **Skipped Pairs:** Remembers pairs of documents that a user has reviewed and decided not to treat as duplicates, preventing them from reappearing in future review sessions.

## Project Structure

```
.
├── backfill_embeddings.py  # Script to generate and store embeddings for documents.
├── deduplicator.py         # Main Streamlit application for finding and managing duplicates.
├── purge_deleted_docs.py   # Script to permanently delete soft-deleted documents and storage files.
├── README.md               # This file.
├── requirements.txt        # Python dependencies for the project.
├── run.sh                  # Shell script to easily start the Streamlit application.
├── setup_venv.sh           # Shell script to set up a Python virtual environment and install dependencies.
└── .env.example            # Example environment file (create a .env from this).
```

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd file_deduplicator
    ```

2.  **Install Doppler CLI:**
    If you haven't already, install the Doppler CLI. Follow the official instructions:
    [https://docs.doppler.com/docs/cli](https://docs.doppler.com/docs/cli)
    After installation, log in to your Doppler account:
    ```bash
    doppler login
    ```

3.  **Set up Doppler for this Project:**
    Navigate to the project directory and run `doppler setup`. This will link this project directory to your desired Doppler project and configuration (e.g., `dev`).
    ```bash
    cd /path/to/file_deduplicator
    doppler setup
    ```
    Follow the prompts to select your organization, project (the one containing `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`), and config.

4.  **Create and Activate Virtual Environment:**
    The `setup_venv.sh` script will create a Python virtual environment (named `venv` by default) and install the required dependencies from `requirements.txt`.
    ```bash
    bash setup_venv.sh
    ```
    After setup, activate the environment:
    ```bash
    source venv/bin/activate
    ```

5.  **Configure Environment Variables:**
    Copy the `.env.example` file to `.env` and fill in your Supabase project details:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and provide:
    *   `SUPABASE_URL`: Your Supabase project URL.
    *   `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase service role key (this is required for operations like updating and deleting data, and for the backfill and purge scripts).
    ```
    Your Supabase secrets (`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`) should be managed in the Doppler project you linked in step 3. The Python scripts will automatically have access to these as environment variables when run via the Doppler CLI.

## Usage

Ensure your virtual environment is activated (`source venv/bin/activate`) and you are in the project directory where Doppler is set up.

All scripts should be run using `doppler run -- your_command` so that secrets are injected.

### 1. Backfill Embeddings (If Needed)

If you have existing documents in your Supabase table that do not have embeddings, or if you want to re-generate embeddings, run the backfill script:
```bash
doppler run -- python backfill_embeddings.py
```
This script will fetch documents missing embeddings, calculate them using the `all-MiniLM-L6-v2` model (by default), and update the records in Supabase.

### 2. Run the Deduplicator Application

To start the Streamlit web application for reviewing duplicates:
```bash
doppler run -- bash run.sh
```
Or directly:
```bash
doppler run -- streamlit run deduplicator.py --server.fileWatcherType none
```
The application will:
*   Connect to your Supabase instance.
*   Fetch documents that have embeddings.
*   Identify pairs of documents with similarity above a defined threshold.
*   Allow you to auto-process "perfect matches" (very high similarity).
*   Present other potential duplicates for manual review. You can compare content, see metadata, and choose to:
    *   Keep one document and soft-delete the other (typically the older one).
    *   Skip the pair if they are not actual duplicates.

### 3. Purge Soft-Deleted Documents

After reviewing and soft-deleting documents, you can permanently remove them from the database and their associated files from Supabase Storage using the `purge_deleted_docs.py` script.

**It is highly recommended to run this with the `--dry-run` flag first to see what would be deleted without actually performing any deletions.**
```bash
doppler run -- python purge_deleted_docs.py --dry-run
```
Once you are confident, run without the flag to perform the purge:
```bash
doppler run -- python purge_deleted_docs.py
```
This script will:
*   Fetch documents marked with `is_deleted = true`.
*   Attempt to delete the corresponding file from Supabase Storage.
*   Log the deletion in a `processed_duplicates` table (if configured).
*   Permanently delete the document row from the `documents` table.

## Key Technologies

*   **Python 3:** Core programming language.
*   **Streamlit:** For building the interactive web application.
*   **Supabase:** Backend-as-a-Service for database (PostgreSQL) and file storage.
    *   **pgvector:** PostgreSQL extension used for storing and querying vector embeddings.
*   **Sentence Transformers:** Library for generating state-of-the-art sentence and text embeddings.
*   **dotenv:** For managing environment variables.

## Configuration Details

### Environment Variables (`.env` file)

*   `SUPABASE_URL`: (Required) The URL of your Supabase project.
*   `SUPABASE_SERVICE_ROLE_KEY`: (Required) The service role API key for your Supabase project. This key has elevated privileges and should be kept secret.

### Secrets Management (Doppler)

This project uses Doppler for managing secrets such as:
*   `SUPABASE_URL`: Your Supabase project URL.
*   `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase service role key.

These secrets are injected as environment variables at runtime by the Doppler CLI. Ensure you have run `doppler setup` in the project's root directory to link it to the correct Doppler project and configuration.
The `.env` file is no longer used.

### Script Configuration

Some scripts have internal configuration variables that can be adjusted:

*   **`deduplicator.py`:**
    *   `TABLE_NAME`: Name of the Supabase table containing documents.
    *   `TEXT_COLUMN`: Column with the main text content.
    *   `EMBEDDING_COLUMN`: Column where embeddings are stored.
    *   `SIMILARITY_THRESHOLD`: Minimum similarity score (0.0 to 1.0) for a pair to be considered a potential duplicate.
    *   `PERFECT_MATCH_THRESHOLD`: Similarity score at or above which pairs can be auto-processed.
*   **`backfill_embeddings.py`:**
    *   `TABLE_NAME`, `TEXT_COLUMN`, `EMBEDDING_COLUMN`: Similar to `deduplicator.py`.
    *   `MODEL_NAME`: The Sentence Transformer model to use (e.g., `all-MiniLM-L6-v2`).
    *   `BATCH_SIZE`: Number of documents to process in each batch.
*   **`purge_deleted_docs.py`:**
    *   `DOCUMENTS_TABLE`: Name of the main documents table.
    *   `PROCESSED_TABLE`: Name of the table to log purged documents (optional, depends on setup).
    *   `STORAGE_BUCKET`: Name of the Supabase Storage bucket where document files are stored.
    *   `--batch-size` (command-line argument): Number of documents to process per batch.

Ensure these configurations match your Supabase setup and requirements.

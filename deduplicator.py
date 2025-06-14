import os
import streamlit as st
from supabase import create_client, Client
from sentence_transformers import util
import numpy as np
import torch
import itertools
from dateutil import parser
import json # Add json import
import subprocess # Add subprocess import
import sys # Add sys import

st.set_page_config(layout="wide") # MUST be the first Streamlit command

# --- Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") # Changed from SUPABASE_SERVICE_ROLE_KEY
TABLE_NAME = "documents" # TODO: Make this configurable or fetch dynamically
TEXT_COLUMN = "extracted_content"   # Column containing the document text (needed for display)
ID_COLUMN = "id"          # Primary key column
CREATED_AT_COLUMN = "last_modified" # Creation timestamp column
IS_DELETED_COLUMN = "is_deleted" # Soft delete flag column
ORIGINAL_FILENAME_COLUMN = "original_filename" # Filename column
FILE_TYPE_COLUMN = "file_type" # File type column
EMBEDDING_COLUMN = "embedding" # The vector column storing pre-computed embeddings
SIMILARITY_THRESHOLD = 0.90 # Adjust as needed (0.0 to 1.0)
PERFECT_MATCH_THRESHOLD = 0.99 # Threshold for automatic deletion (similarity >= this value)

# --- Supabase Client ---
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        st.sidebar.success("Connected to Supabase!")
    except Exception as e:
        st.sidebar.error(f"Failed to connect to Supabase: {e}")
        st.stop()
else:
    st.sidebar.error("Supabase URL or Key not found in environment variables.")
    st.info("Please create a `.env` file with SUPABASE_URL and SUPABASE_KEY.")
    st.stop()

# --- Utility Functions ---
@st.cache_data
def check_missing_embeddings(_supabase_client: Client):
    """Counts non-deleted documents with NULL embeddings."""
    try:
        response = (_supabase_client.table(TABLE_NAME)
                    .select("id", count='exact')
                    .eq(IS_DELETED_COLUMN, False)
                    .is_(EMBEDDING_COLUMN, 'null')
                    .execute())
        return response.count if response.count is not None else 0
    except Exception as e:
        st.sidebar.error(f"Error checking for missing embeddings: {e}")
        return -1 # Indicate error

# --- Data Fetching ---
@st.cache_data # Re-enable cache
def fetch_documents_with_embeddings(_supabase_client: Client):
    """
    Fetches non-deleted documents with pre-computed embeddings from Supabase.
    Selects only documents that HAVE an embedding.
    """
    documents_with_embeddings = []
    offset = 0
    limit = 1000 # Supabase default limit per query

    st.sidebar.write("Fetching documents with embeddings...")
    progress_bar = st.sidebar.progress(0)
    total_fetched = 0
    estimated_total = -1 # Flag to fetch count once

    try:
        select_columns = f"{ID_COLUMN}, {CREATED_AT_COLUMN}, {ORIGINAL_FILENAME_COLUMN}, {FILE_TYPE_COLUMN}, {EMBEDDING_COLUMN}"

        while True:
            # Estimate total count on first iteration for better progress bar
            if estimated_total == -1:
                 try:
                    count_response = _supabase_client.table(TABLE_NAME).select(ID_COLUMN, count='exact').not_.is_(EMBEDDING_COLUMN, 'null').eq(IS_DELETED_COLUMN, False).execute()
                    estimated_total = count_response.count if count_response.count is not None else 0
                    st.sidebar.write(f"Expecting approx {estimated_total} documents...")
                 except Exception as count_e:
                     st.sidebar.warning(f"Could not estimate total count: {count_e}")
                     estimated_total = 0 # Avoid retrying count

            response = (_supabase_client.table(TABLE_NAME)
                        .select(select_columns)
                        .not_.is_(EMBEDDING_COLUMN, 'null') # Only fetch rows with embeddings
                        .eq(IS_DELETED_COLUMN, False)      # Only non-deleted
                        .range(offset, offset + limit - 1)
                        .execute())

            if response.data:
                documents_with_embeddings.extend(response.data)
                total_fetched = len(documents_with_embeddings) # Use len() for accurate count
                st.sidebar.info(f"Fetched {total_fetched} documents...")
                # Update progress bar
                if estimated_total > 0:
                    progress_bar.progress(min(1.0, total_fetched / estimated_total))
                else: # Fallback progress if count failed
                    progress_bar.progress((offset // limit) % 100 / 100.0)

                # Check if this was the last page
                if len(response.data) < limit:
                    break
                offset += limit
            else:
                # No more data or initial fetch failed
                break

        progress_bar.empty() # Remove progress bar after completion/error
        st.sidebar.success(f"Finished fetching {len(documents_with_embeddings)} documents with embeddings.")
        if not documents_with_embeddings:
             st.sidebar.warning(f"No documents with embeddings found in table '{TABLE_NAME}'. Did the backfill script run?")

        return documents_with_embeddings

    except Exception as e:
        progress_bar.empty()
        st.error(f"Error fetching documents with embeddings: {e}")
        return []

@st.cache_data # Re-enable cache
def find_similar_pairs(docs_with_embeddings, threshold, skipped_pairs_set):
    """
    Finds pairs of documents with similarity above the threshold using pre-computed embeddings.
    Excludes skipped pairs.
    """
    similar_pairs = []
    if not docs_with_embeddings or len(docs_with_embeddings) < 2:
        return similar_pairs

    st.info(f"Comparing {len(docs_with_embeddings)} documents...")

    # Extract embeddings and convert them to a tensor
    # Embeddings from pgvector are typically lists
    try:
        embeddings_list = []
        valid_doc_indices = [] # Keep track of indices corresponding to valid embeddings

        # --- REMOVED DEBUGGING --- 
        # st.write("--- Checking embedding types ---")
        # printed_count = 0
        # --- /REMOVED DEBUGGING --- 

        for i, doc in enumerate(docs_with_embeddings):
             embedding_raw = doc.get(EMBEDDING_COLUMN)
             parsed_embedding = None

             # --- REMOVED DEBUGGING --- 
             # if printed_count < 5: ...
             # --- /REMOVED DEBUGGING --- 

             if isinstance(embedding_raw, list):
                 # Ideal case: Already a list
                 parsed_embedding = embedding_raw
             elif isinstance(embedding_raw, str):
                 # Workaround: Try parsing the string
                 try:
                     parsed_list = json.loads(embedding_raw)
                     if isinstance(parsed_list, list):
                         parsed_embedding = parsed_list
                     else:
                         st.warning(f"Parsed string for Doc ID {doc.get(ID_COLUMN, 'N/A')} but result was not a list: type {type(parsed_list)}")
                 except json.JSONDecodeError:
                     st.warning(f"Could not JSON decode embedding string for Doc ID {doc.get(ID_COLUMN, 'N/A')}: Value starts with '{embedding_raw[:20]}...'")
                 except Exception as e:
                      st.warning(f"Unexpected error parsing embedding string for Doc ID {doc.get(ID_COLUMN, 'N/A')}: {e}")

             # Check if we have a valid list embedding after parsing attempts
             if isinstance(parsed_embedding, list):
                 embeddings_list.append(parsed_embedding)
                 valid_doc_indices.append(i)
             else:
                 # Log if it wasn't a list or a successfully parsed string
                 if embedding_raw is not None: # Avoid logging for genuinely NULL embeddings 
                    st.warning(f"Skipping document ID {doc.get(ID_COLUMN, 'N/A')} due to invalid/unparsable embedding: type {type(embedding_raw)}")

        if not embeddings_list:
             st.warning("No valid embeddings found in fetched documents to compare.")
             return []

        # Convert list of lists to PyTorch Tensor
        embeddings_tensor = torch.tensor(embeddings_list, dtype=torch.float)
        st.info(f"Converted {len(embeddings_list)} valid embeddings to tensor for comparison.")

    except Exception as e:
        st.error(f"Error processing embeddings into tensor: {e}")
        return []


    # Calculate cosine similarity between all pairs of *valid* embeddings
    with st.spinner("Calculating similarity scores..."):
        try:
            # This computes similarity only between the valid embeddings
            cosine_scores = util.cos_sim(embeddings_tensor, embeddings_tensor)
        except Exception as e:
             st.error(f"Error calculating cosine similarity: {e}")
             return []

    # Find pairs with similarity above threshold
    # Iterate through the indices corresponding to the cosine_scores matrix
    st.info("Identifying similar pairs above threshold...")
    with st.spinner("Filtering pairs..."):
        num_valid = len(valid_doc_indices)
        for i_idx in range(num_valid):
            for j_idx in range(i_idx + 1, num_valid):
                 similarity_score = cosine_scores[i_idx][j_idx].item()

                 # Check 1: Similarity threshold
                 if similarity_score >= threshold:
                    # Map back to original document indices
                    original_i = valid_doc_indices[i_idx]
                    original_j = valid_doc_indices[j_idx]

                    doc1_meta = docs_with_embeddings[original_i]
                    doc2_meta = docs_with_embeddings[original_j]

                    # Check 2: File types match
                    doc1_type = doc1_meta.get(FILE_TYPE_COLUMN)
                    doc2_type = doc2_meta.get(FILE_TYPE_COLUMN)

                    if doc1_type is not None and doc1_type == doc2_type:
                        # Check 3: Pair not skipped
                        id1 = doc1_meta[ID_COLUMN]
                        id2 = doc2_meta[ID_COLUMN]
                        ordered_pair = (min(id1, id2), max(id1, id2))

                        if ordered_pair not in skipped_pairs_set:
                            # Add pair - note: doc1/doc2 here DON'T contain text yet
                            similar_pairs.append({
                                "doc1_meta": doc1_meta, # Store metadata + embedding
                                "doc2_meta": doc2_meta, # Store metadata + embedding
                                "similarity": similarity_score
                            })

    st.info(f"Found {len(similar_pairs)} raw similar pairs before sorting.")
    # Sort pairs by similarity score (descending)
    similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return similar_pairs

# --- Database Update ---
def soft_delete_document(_supabase_client: Client, doc_id):
    """Marks a document as deleted in Supabase."""
    try:
        _supabase_client.table(TABLE_NAME) \
            .update({IS_DELETED_COLUMN: True}) \
            .eq(ID_COLUMN, doc_id) \
            .execute()
        # Consider clearing relevant caches if needed after delete
        # Specifically, fetch_documents_with_embeddings might need clearing
        fetch_documents_with_embeddings.clear()
        return True
    except Exception as e:
        st.error(f"Error soft deleting document ID {doc_id}: {e}")
        return False

# --- Skipped Pair Handling ---
SKIPPED_PAIRS_TABLE = "skipped_pairs"

@st.cache_data
def fetch_skipped_pairs(_supabase_client: Client):
    """Fetches the set of skipped document ID pairs from the database."""
    skipped_set = set()
    try:
        # Fetch all skipped pairs - pagination might be needed for huge skip lists
        response = _supabase_client.table(SKIPPED_PAIRS_TABLE).select("doc_id_1, doc_id_2").execute()
        if response.data:
            for pair in response.data:
                id1 = pair['doc_id_1']
                id2 = pair['doc_id_2']
                skipped_set.add((min(id1, id2), max(id1, id2)))
            st.sidebar.info(f"Fetched {len(skipped_set)} skipped pairs.")
        else:
             st.sidebar.info("No previously skipped pairs found.")
    except Exception as e:
        st.sidebar.warning(f"Could not fetch skipped pairs: {e}")
    return skipped_set

def add_skipped_pair(_supabase_client: Client, doc_id1, doc_id2):
    """Adds a pair of document IDs to the skipped_pairs table."""
    id1, id2 = (doc_id1, doc_id2) if doc_id1 < doc_id2 else (doc_id2, doc_id1)
    try:
        _supabase_client.table(SKIPPED_PAIRS_TABLE).insert({
            "doc_id_1": id1,
            "doc_id_2": id2
        }).execute()
        # No cache clearing needed here now for fetch_skipped_pairs
        return True
    except Exception as e:
        st.error(f"Error adding skipped pair ({id1}, {id2}): {e}")
        if "duplicate key value violates unique constraint" in str(e).lower():
             st.warning(f"Pair ({id1}, {id2}) already marked as skipped.")
             return True
        return False

# --- Function to fetch text content for display ---
# @st.cache_data # Cache this carefully - might hide recent updates if not cleared
def get_document_content(_supabase_client: Client, doc_id):
    """Fetches the text content of a single document by ID."""
    try:
        response = _supabase_client.table(TABLE_NAME).select(TEXT_COLUMN).eq(ID_COLUMN, doc_id).maybe_single().execute()
        if response.data:
            return response.data.get(TEXT_COLUMN, "[Content not found in DB]")
        else:
            # This case should be less likely if the ID came from fetch_documents_with_embeddings
            st.warning(f"Document ID {doc_id} not found when fetching content.")
            return "[Content not found]"
    except Exception as e:
        st.error(f"Error fetching content for ID {doc_id}: {e}")
        return "[Error fetching content]"

# --- UI Display ---
def display_duplicates(pairs, skipped_pairs_set): # Receives pairs with 'docX_meta' keys
    """Displays duplicate pairs and handles user action."""
    st.header("Potential Duplicates for Manual Review") # Clarify header
    if not pairs:
        st.info("No duplicates found or remaining to review in this session.")
        return

    # Ensure session state for processed pairs is initialized
    if 'processed_pairs_tuple' not in st.session_state:
        st.session_state.processed_pairs_tuple = set()

    # Find the next pair index to display whose ID tuple hasn't been processed *in this session*
    pair_to_display = None
    pair_index_in_current_list = -1 # Keep track of index within the *filtered* `pairs` list
    for i, pair_candidate in enumerate(pairs):
        id1 = pair_candidate["doc1_meta"][ID_COLUMN]
        id2 = pair_candidate["doc2_meta"][ID_COLUMN]
        current_pair_tuple = (min(id1, id2), max(id1, id2))
        
        if current_pair_tuple not in st.session_state.processed_pairs_tuple:
            pair_to_display = pair_candidate
            pair_index_in_current_list = i # Store the index relative to the current list `pairs`
            break

    if pair_to_display is None:
        st.success("All potential duplicate pairs for this session have been processed!")
        if st.button("Re-fetch and Re-check All?"):
            # Clear specific session state related to processing
            if 'processed_pairs_tuple' in st.session_state: del st.session_state['processed_pairs_tuple']
            if 'auto_deleted_ids' in st.session_state: del st.session_state['auto_deleted_ids'] 
            st.cache_data.clear() # Clear all data caches
            st.rerun()
        return

    # Use the found pair
    pair = pair_to_display
    doc1_meta = pair["doc1_meta"]
    doc2_meta = pair["doc2_meta"]
    similarity = pair["similarity"]

    # Calculate the display number correctly based on total pairs and processed ones
    total_review_pairs = len(pairs)
    processed_count = len(st.session_state.processed_pairs_tuple)
    current_review_number = processed_count + 1 # The number of the item being reviewed
    st.subheader(f"Reviewing Pair {current_review_number} (Similarity: {similarity:.2f}) - Index {pair_index_in_current_list}")

    # Determine older document based on metadata
    try:
        # Ensure timestamps are valid before parsing
        ts1_str = doc1_meta.get(CREATED_AT_COLUMN)
        ts2_str = doc2_meta.get(CREATED_AT_COLUMN)
        if ts1_str is None or ts2_str is None:
             raise ValueError("Missing timestamp for comparison")
        ts1 = parser.isoparse(ts1_str)
        ts2 = parser.isoparse(ts2_str)
        older_doc_meta = doc1_meta if ts1 < ts2 else doc2_meta
        newer_doc_meta = doc2_meta if ts1 < ts2 else doc1_meta
    except (TypeError, ValueError, KeyError) as e:
        st.warning(f"Could not reliably determine older document due to timestamp issue: {e}. Defaulting order.")
        older_doc_meta = doc1_meta
        newer_doc_meta = doc2_meta
    older_doc_id = older_doc_meta[ID_COLUMN]
    newer_doc_id = newer_doc_meta[ID_COLUMN]

    # Fetch content only for the pair being displayed
    with st.spinner(f"Loading content for Docs {doc1_meta[ID_COLUMN]} & {doc2_meta[ID_COLUMN]}..."):
        doc1_content = get_document_content(supabase, doc1_meta[ID_COLUMN])
        doc2_content = get_document_content(supabase, doc2_meta[ID_COLUMN])

    col1, col2 = st.columns(2)

    # Define the tuple for the current pair
    current_pair_tuple = (min(doc1_meta[ID_COLUMN], doc2_meta[ID_COLUMN]), max(doc1_meta[ID_COLUMN], doc2_meta[ID_COLUMN]))

    with col1:
        st.markdown(f"**Document 1 (ID: {doc1_meta[ID_COLUMN]})**")
        st.markdown(f"*Filename: {doc1_meta.get(ORIGINAL_FILENAME_COLUMN, 'N/A')}*")
        st.markdown(f"*Type: {doc1_meta.get(FILE_TYPE_COLUMN, 'N/A')}*")
        st.markdown(f"*Created: {doc1_meta.get(CREATED_AT_COLUMN, 'N/A')}*")
        if doc1_meta[ID_COLUMN] == older_doc_id:
            st.warning("(Potential Older Version)")
        else:
            st.markdown("&nbsp;", unsafe_allow_html=True)
        st.text_area("Content 1", doc1_content, height=300, key=f"text1_{pair_index_in_current_list}")

        is_newer = (doc1_meta[ID_COLUMN] == newer_doc_id)
        if st.button(f"Keep this, Delete Older (ID: {older_doc_id})", key=f"keep1_del_older_{pair_index_in_current_list}", disabled=not is_newer):
            if soft_delete_document(supabase, older_doc_id):
                st.success(f"Soft deleted document ID: {older_doc_id}")
                # Add the processed pair tuple to session state
                st.session_state.processed_pairs_tuple.add(current_pair_tuple)
                st.rerun()
            else:
                st.error("Soft deletion failed.")

    with col2:
        st.markdown(f"**Document 2 (ID: {doc2_meta[ID_COLUMN]})**")
        st.markdown(f"*Filename: {doc2_meta.get(ORIGINAL_FILENAME_COLUMN, 'N/A')}*")
        st.markdown(f"*Type: {doc2_meta.get(FILE_TYPE_COLUMN, 'N/A')}*")
        st.markdown(f"*Created: {doc2_meta.get(CREATED_AT_COLUMN, 'N/A')}*")
        if doc2_meta[ID_COLUMN] == older_doc_id:
            st.warning("(Potential Older Version)")
        else:
            st.markdown("&nbsp;", unsafe_allow_html=True)
        st.text_area("Content 2", doc2_content, height=300, key=f"text2_{pair_index_in_current_list}")

        is_newer = (doc2_meta[ID_COLUMN] == newer_doc_id)
        if st.button(f"Keep this, Delete Older (ID: {older_doc_id})", key=f"keep2_del_older_{pair_index_in_current_list}", disabled=not is_newer):
             if soft_delete_document(supabase, older_doc_id):
                st.success(f"Soft deleted document ID: {older_doc_id}")
                # Add the processed pair tuple to session state
                st.session_state.processed_pairs_tuple.add(current_pair_tuple)
                st.rerun()
             else:
                st.error("Soft deletion failed.")

    st.markdown("---")

    if st.button("Skip this pair", key=f"skip_{pair_index_in_current_list}"):
        if add_skipped_pair(supabase, doc1_meta[ID_COLUMN], doc2_meta[ID_COLUMN]):
            st.success(f"Marked pair ({doc1_meta[ID_COLUMN]}, {doc2_meta[ID_COLUMN]}) as skipped in database.")
            # Also add to session state to advance UI
            st.session_state.processed_pairs_tuple.add(current_pair_tuple)
            st.rerun()
        else:
            st.error("Failed to mark pair as skipped in the database.")

# Helper function to determine older document from metadata
# Returns the ID of the older document, or None if determination fails
def get_older_doc_id(doc1_meta, doc2_meta):
    try:
        ts1_str = doc1_meta.get(CREATED_AT_COLUMN)
        ts2_str = doc2_meta.get(CREATED_AT_COLUMN)
        if ts1_str is None or ts2_str is None:
            st.warning(f"Missing timestamp for comparison between {doc1_meta.get(ID_COLUMN)} and {doc2_meta.get(ID_COLUMN)}. Cannot auto-delete.")
            return None
        ts1 = parser.isoparse(ts1_str)
        ts2 = parser.isoparse(ts2_str)
        return doc1_meta[ID_COLUMN] if ts1 < ts2 else doc2_meta[ID_COLUMN]
    except (TypeError, ValueError, KeyError) as e:
        st.warning(f"Could not parse timestamps for comparison between {doc1_meta.get(ID_COLUMN)} and {doc2_meta.get(ID_COLUMN)}: {e}. Cannot auto-delete.")
        return None

# --- Main Application Logic ---
st.title("Supabase Document Deduplicator")

if supabase:
    # Display config settings
    st.sidebar.title("Configuration")
    st.sidebar.write(f"Table: `{TABLE_NAME}`")
    st.sidebar.write(f"Embedding Column: `{EMBEDDING_COLUMN}`")
    st.sidebar.write(f"Similarity Threshold: `{SIMILARITY_THRESHOLD}`")
    st.sidebar.write(f"Perfect Match >= `{PERFECT_MATCH_THRESHOLD}`")

    # Initialize session state
    if 'processed_pairs_tuple' not in st.session_state:
         st.session_state.processed_pairs_tuple = set()
    if 'auto_deleted_ids' not in st.session_state:
        st.session_state.auto_deleted_ids = set() # Store IDs deleted in this run

    # Check and display missing embeddings
    missing_count = check_missing_embeddings(supabase)
    if missing_count > 0:
        st.sidebar.warning(f"{missing_count} documents are missing embeddings.")
        if st.sidebar.button("Generate Missing Embeddings"):
            st.sidebar.info("Starting embedding generation... This might take a while.")
            with st.spinner("Running backfill_embeddings.py..."):
                try:
                    # Ensure using the same python interpreter
                    process = subprocess.run(
                        [sys.executable, "backfill_embeddings.py"],
                        capture_output=True,
                        text=True,
                        check=False # Don't raise exception on non-zero exit
                    )
                    st.sidebar.subheader("Backfill Script Output:")
                    st.sidebar.code(process.stdout + "\n" + process.stderr)
                    if process.returncode == 0:
                        st.sidebar.success("Embedding generation finished successfully!")
                        # Clear caches to force re-fetch
                        check_missing_embeddings.clear()
                        fetch_documents_with_embeddings.clear()
                        find_similar_pairs.clear() 
                        st.rerun() # Rerun the app to reflect changes
                    else:
                        st.sidebar.error(f"Embedding generation failed with exit code {process.returncode}.")
                except Exception as e:
                    st.sidebar.error(f"Failed to run backfill script: {e}")
    elif missing_count == 0:
        st.sidebar.success("All documents have embeddings.")
    else: # missing_count == -1
        st.sidebar.error("Could not determine missing embedding count.")

    st.sidebar.markdown("--- ") # Separator

    # Other sidebar elements
    threshold = st.sidebar.slider("Similarity Threshold", 0.7, 1.0, 0.9, 0.01)
    items_per_page = st.sidebar.selectbox("Items per Page", [10, 20, 50, 100], index=1)

    # Fetch documents (metadata + embeddings) and skipped pairs
    # These are cached, so only fetched once per session unless cache cleared
    docs_with_embeddings = fetch_documents_with_embeddings(supabase)
    skipped_pairs_set = fetch_skipped_pairs(supabase)

    if docs_with_embeddings:
        # No embedding calculation step needed here

        # Find *all* similar pairs using pre-computed embeddings
        all_similar_pairs = find_similar_pairs(docs_with_embeddings, SIMILARITY_THRESHOLD, skipped_pairs_set)

        # --- Partition into Perfect Matches and Review Candidates ---
        perfect_matches = []
        review_candidates = []
        if all_similar_pairs:
            st.info(f"Found {len(all_similar_pairs)} total similar pairs above {SIMILARITY_THRESHOLD} threshold.")
            for pair in all_similar_pairs:
                # Round the similarity score before comparing for perfect match
                # Use a reasonable precision, e.g., 6 decimal places
                rounded_similarity = round(pair["similarity"], 6)
                if rounded_similarity >= PERFECT_MATCH_THRESHOLD:
                    perfect_matches.append(pair)
                else:
                    review_candidates.append(pair)
            st.info(f"Partitioned into {len(perfect_matches)} perfect matches (>= {PERFECT_MATCH_THRESHOLD} after rounding) and {len(review_candidates)} candidates for review.")

        else:
            st.info("No similar pairs found above the threshold.")

        # --- Auto-process Perfect Matches ---
        if perfect_matches and st.sidebar.button("⚡️ Auto-process Perfect Matches"):
            ids_to_delete = set();
            processed_perfect_pairs = 0;
            with st.spinner(f"Identifying older documents in {len(perfect_matches)} perfect matches..."):
                for pair_idx, pair in enumerate(perfect_matches):
                    older_id = get_older_doc_id(pair["doc1_meta"], pair["doc2_meta"])
                    if older_id:
                        ids_to_delete.add(older_id);
                    # We consider the pair processed even if older couldn't be determined
                    processed_perfect_pairs += 1;
            
            st.info(f"Identified {len(ids_to_delete)} unique older documents to delete from {processed_perfect_pairs} perfect matches.");

            if ids_to_delete:
                auto_deleted_count = 0;
                failed_deletions = [];
                delete_progress = st.progress(0);
                with st.spinner(f"Attempting to auto-delete {len(ids_to_delete)} older documents..."):
                    for i, older_id in enumerate(ids_to_delete):
                        if soft_delete_document(supabase, older_id):
                            auto_deleted_count += 1;
                            st.session_state.auto_deleted_ids.add(older_id) # Track success
                        else:
                            failed_deletions.append(older_id);
                        delete_progress.progress((i + 1) / len(ids_to_delete));
                delete_progress.empty();

                if auto_deleted_count > 0:
                    st.success(f"✅ Automatically soft-deleted {auto_deleted_count} older documents from perfect matches.");
                    # Clear cache to reflect deletions in subsequent steps/runs
                    st.cache_data.clear();
                    # Rerun only if auto-delete happened to refresh the state cleanly
                    st.info("Refreshing data after auto-deletion...");
                    st.rerun(); 

                if failed_deletions:
                    st.error(f"❌ Failed to auto-delete {len(failed_deletions)} documents: {failed_deletions}");
            else:
                 st.warning("Could not identify any older documents to delete automatically (check timestamps?).");

        
        # Filter out pairs where one document was *just* auto-deleted in this run
        # This avoids showing pairs that are no longer relevant after auto-processing
        final_review_candidates = [
            pair for pair in review_candidates 
            if pair["doc1_meta"][ID_COLUMN] not in st.session_state.auto_deleted_ids and 
               pair["doc2_meta"][ID_COLUMN] not in st.session_state.auto_deleted_ids
        ]
        
        # Display pairs for manual review
        if final_review_candidates:
            display_duplicates(final_review_candidates, skipped_pairs_set)
        else:
            st.info("No pairs remaining for manual review.")

    else:
        st.warning("No documents with embeddings found or fetched. Please run the `backfill_embeddings.py` script.")

else:
    st.error("Supabase client not initialized.")
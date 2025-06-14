#!/bin/bash
# Simple script to run the Streamlit deduplicator app

echo "Starting the Document Deduplicator (via run.sh using 'command streamlit')..."
# Secrets should be injected by the outer 'doppler run -- ./run.sh' call
# Using 'command streamlit' to bypass any potential shell aliases or functions
command streamlit run deduplicator.py --server.fileWatcherType none
echo "run.sh: Streamlit command has been executed. If Streamlit started correctly, it will be running. If it exited, this script will now exit."
#!/bin/bash
# Simple script to run the Streamlit deduplicator app

echo "Starting the Document Deduplicator..."
# Secrets should be injected by the outer 'doppler run -- ./run.sh' call
doppler run -- ./run.sh
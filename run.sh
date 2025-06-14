#!/bin/bash
# Simple script to run the Streamlit deduplicator app

echo "Starting the Document Deduplicator via Doppler..."
# Add --server.fileWatcherType none to potentially avoid torch introspection issues
doppler run -- streamlit run deduplicator.py --server.fileWatcherType none
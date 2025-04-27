#!/bin/bash
# Simple script to run the Streamlit deduplicator app

echo "Starting the Document Deduplicator..."
# Add --server.fileWatcherType none to potentially avoid torch introspection issues
streamlit run deduplicator.py --server.fileWatcherType none 
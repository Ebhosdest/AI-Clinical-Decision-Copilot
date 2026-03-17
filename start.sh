#!/bin/bash
set -e

if [ ! -d "data/vector_index" ]; then
    echo "Vector index not found. Building..."
    python -m src.rag.pipeline build
fi

exec streamlit run frontend/app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true

#!/bin/bash
set -e

# Set Python path
export PYTHONPATH="/app"

# Run the application
exec uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

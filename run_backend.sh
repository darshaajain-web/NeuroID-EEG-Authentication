#!/bin/zsh
cd "$(dirname "$0")/backend"
python3 -m uvicorn app:app --reload --host 127.0.0.1 --port 8000

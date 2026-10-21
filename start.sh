#!/usr/bin/env sh
cd "$(dirname "$0")" || exit 1
if [ ! -d .venv ]; then
    echo "Setting up for the first time..."
    python3 -m venv .venv
    . .venv/bin/activate
    pip install -e .
else
    . .venv/bin/activate
fi
python -m ticket_classifier

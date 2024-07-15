#!/usr/bin/env bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

export CUDA_VISIBLE_DEVICES=0

jq -c '.[]' 'setup_cfg.json' | while read cfg; do
    SESSION_NAME="$(echo $cfg | jq -r '.session_name')"
    if [ -d "$SESSION_NAME/spacy" ]; then
        echo "Folder $SESSION_NAME/spacy already found."
        continue
    fi
    if [ ! -f "$SESSION_NAME/dataset_applied.jsonl" ]; then
        echo "Session $SESSION_NAME has no imputed dataset so far."
        continue
    fi

    JSONL_FILE="$SESSION_NAME/dataset_applied.jsonl"
    SPACY_OUTPUT="$SESSION_NAME/spacy"

    echo "Try to start retraining for $SESSION_NAME at $(date +"%Y-%m-%d %T")..." | tee -a "$SESSION_NAME/retrain.log"
    python3 wikionto_spacy.py train \
        "$JSONL_FILE" \
        "$SPACY_OUTPUT" \
        -n 128  \
        -m "GerMedBERT/medbert-512" 2>&1 | tee -a "$SESSION_NAME/apply.log"
    echo "Stopping retraining for $SESSION_NAME at $(date +"%Y-%m-%d %T")..." | tee -a "$SESSION_NAME/retrain.log"
done
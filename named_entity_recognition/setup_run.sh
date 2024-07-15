#!/usr/bin/env bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

export CUDA_VISIBLE_DEVICES=0

jq -c '.[]' 'setup_cfg.json' | while read cfg; do
    SESSION_NAME="$(echo $cfg | jq -r '.session_name')"
    if [ -d "$SESSION_NAME" ]; then
        echo "Folder $SESSION_NAME already exists."
        continue
    fi
    mkdir -p "$SESSION_NAME"
    echo "Try to start session $SESSION_NAME at $(date +"%Y-%m-%d %T")..." | tee -a "$SESSION_NAME/train.log"
    python3 wikionto_train.py "$cfg" 2>&1 | tee -a "$SESSION_NAME/train.log"

    # Save setup configuration and dataset
    cp "$(echo $cfg | jq -r '.dataset_file')" "$SESSION_NAME/dataset.jsonl"
    echo $cfg | jq . > "$SESSION_NAME/cfg.json"

    echo "Stopping session $SESSION_NAME at $(date +"%Y-%m-%d %T")..." | tee -a "$SESSION_NAME/train.log"
done
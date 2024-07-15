#!/usr/bin/env bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

export CUDA_VISIBLE_DEVICES=0

jq -c '.[]' 'setup_cfg.json' | while read cfg; do
    SESSION_NAME="$(echo $cfg | jq -r '.session_name')"
    if [ ! -d "$SESSION_NAME/results/checkpoint-best" ]; then
        echo "Folder $SESSION_NAME/results/checkpoint-best could not be found."
        continue
    fi
    if [ -f "$SESSION_NAME/dataset_applied.jsonl" ]; then
        echo "Session $SESSION_NAME already has an imputed dataset."
        continue
    fi
    echo "Try to start dataset imputation for $SESSION_NAME at $(date +"%Y-%m-%d %T")..." | tee -a "$SESSION_NAME/apply.log"
    python3 wikionto_apply.py "$cfg" 2>&1 | tee -a "$SESSION_NAME/apply.log"
    echo "Stopping dataset imputation for $SESSION_NAME at $(date +"%Y-%m-%d %T")..." | tee -a "$SESSION_NAME/apply.log"
done
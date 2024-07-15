#!/usr/bin/env bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

export CUDA_VISIBLE_DEVICES=0

annotateAndEvaluate(){
    PRED_FOLDER="$1"
    PRED_LMAP="$2"
    EVAL_FILE="$3"
    EVAL_LMAP="$4"

    EVAL_NAME="$(basename -- $EVAL_FILE)"
    PRED_OUT="$PRED_FOLDER/applied_$EVAL_NAME"

    # Generate dataset file first
    if [ ! -f "$PRED_OUT" ]; then
        python3 wikionto_spacy.py predict "$EVAL_FILE" "$PRED_OUT" "$PRED_FOLDER"
    else
        echo "Already found file: $PRED_OUT"
    fi

    # Run evaluation
    PRED_SCORE_OUT="$PRED_FOLDER/scores_$EVAL_NAME"
    if [ ! -f "$PRED_SCORE_OUT" ]; then
        python3 wikionto_evaluate.py \
            "$EVAL_FILE" "$EVAL_LMAP" \
            "$PRED_OUT" "$PRED_LMAP" \
            -l 'de' > "$PRED_SCORE_OUT"
    else
        echo "Already found file: $PRED_SCORE_OUT"
    fi
}

# Define evaluation setups
### BRONCO
annotateAndEvaluate "ATC_0.01" '{"ATC_CODE": "Drug"}' "evaluation_datasets/bronco.jsonl" '{"MEDICATION": "Drug"}'
annotateAndEvaluate "ATC_0.05" '{"ATC_CODE": "Drug"}' "evaluation_datasets/bronco.jsonl" '{"MEDICATION": "Drug"}'
annotateAndEvaluate "ATC_0.1"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/bronco.jsonl" '{"MEDICATION": "Drug"}'
annotateAndEvaluate "ATC_0.2"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/bronco.jsonl" '{"MEDICATION": "Drug"}'
annotateAndEvaluate "ATC_0.5"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/bronco.jsonl" '{"MEDICATION": "Drug"}'
annotateAndEvaluate "ATC_0.8"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/bronco.jsonl" '{"MEDICATION": "Drug"}'
annotateAndEvaluate "ATC_1.0"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/bronco.jsonl" '{"MEDICATION": "Drug"}'

### CARDIO
annotateAndEvaluate "ATC_0.01" '{"ATC_CODE": "Drug"}' "evaluation_datasets/CARDIODE400_main.jsonl" '{"DRUG": "Drug", "ACTIVEING": "Drug"}'
annotateAndEvaluate "ATC_0.05" '{"ATC_CODE": "Drug"}' "evaluation_datasets/CARDIODE400_main.jsonl" '{"DRUG": "Drug", "ACTIVEING": "Drug"}'
annotateAndEvaluate "ATC_0.1"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/CARDIODE400_main.jsonl" '{"DRUG": "Drug", "ACTIVEING": "Drug"}'
annotateAndEvaluate "ATC_0.2"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/CARDIODE400_main.jsonl" '{"DRUG": "Drug", "ACTIVEING": "Drug"}'
annotateAndEvaluate "ATC_0.5"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/CARDIODE400_main.jsonl" '{"DRUG": "Drug", "ACTIVEING": "Drug"}'
annotateAndEvaluate "ATC_0.8"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/CARDIODE400_main.jsonl" '{"DRUG": "Drug", "ACTIVEING": "Drug"}'
annotateAndEvaluate "ATC_1.0"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/CARDIODE400_main.jsonl" '{"DRUG": "Drug", "ACTIVEING": "Drug"}'

### GPTNERMED
annotateAndEvaluate "ATC_0.01" '{"ATC_CODE": "Drug"}' "evaluation_datasets/gptnermed.jsonl" '{"Medikation": "Drug"}'
annotateAndEvaluate "ATC_0.05" '{"ATC_CODE": "Drug"}' "evaluation_datasets/gptnermed.jsonl" '{"Medikation": "Drug"}'
annotateAndEvaluate "ATC_0.1"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/gptnermed.jsonl" '{"Medikation": "Drug"}'
annotateAndEvaluate "ATC_0.2"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/gptnermed.jsonl" '{"Medikation": "Drug"}'
annotateAndEvaluate "ATC_0.5"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/gptnermed.jsonl" '{"Medikation": "Drug"}'
annotateAndEvaluate "ATC_0.8"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/gptnermed.jsonl" '{"Medikation": "Drug"}'
annotateAndEvaluate "ATC_1.0"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/gptnermed.jsonl" '{"Medikation": "Drug"}'

### GERNERMED++
annotateAndEvaluate "ATC_0.01" '{"ATC_CODE": "Drug"}' "evaluation_datasets/gernermedpp.jsonl" '{"Drug": "Drug"}'
annotateAndEvaluate "ATC_0.05" '{"ATC_CODE": "Drug"}' "evaluation_datasets/gernermedpp.jsonl" '{"Drug": "Drug"}'
annotateAndEvaluate "ATC_0.1"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/gernermedpp.jsonl" '{"Drug": "Drug"}'
annotateAndEvaluate "ATC_0.2"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/gernermedpp.jsonl" '{"Drug": "Drug"}'
annotateAndEvaluate "ATC_0.5"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/gernermedpp.jsonl" '{"Drug": "Drug"}'
annotateAndEvaluate "ATC_0.8"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/gernermedpp.jsonl" '{"Drug": "Drug"}'
annotateAndEvaluate "ATC_1.0"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/gernermedpp.jsonl" '{"Drug": "Drug"}'

### GGPONC 2.0 (fine, short)
annotateAndEvaluate "ATC_0.01" '{"ATC_CODE": "Drug"}' "evaluation_datasets/ggponc2_train+dev+test_fine_short.jsonl" '{"Clinical_Drug": "Drug"}'
annotateAndEvaluate "ATC_0.05" '{"ATC_CODE": "Drug"}' "evaluation_datasets/ggponc2_train+dev+test_fine_short.jsonl" '{"Clinical_Drug": "Drug"}'
annotateAndEvaluate "ATC_0.1"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/ggponc2_train+dev+test_fine_short.jsonl" '{"Clinical_Drug": "Drug"}'
annotateAndEvaluate "ATC_0.2"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/ggponc2_train+dev+test_fine_short.jsonl" '{"Clinical_Drug": "Drug"}'
annotateAndEvaluate "ATC_0.5"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/ggponc2_train+dev+test_fine_short.jsonl" '{"Clinical_Drug": "Drug"}'
annotateAndEvaluate "ATC_0.8"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/ggponc2_train+dev+test_fine_short.jsonl" '{"Clinical_Drug": "Drug"}'
annotateAndEvaluate "ATC_1.0"  '{"ATC_CODE": "Drug"}' "evaluation_datasets/ggponc2_train+dev+test_fine_short.jsonl" '{"Clinical_Drug": "Drug"}'
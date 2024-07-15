#!/usr/bin/env python3
import json
from bratevalwrapper4nlp import evaluate
from dseqmap4nlp import SpacySequenceMapper, LabelLoader
import spacy
import torch
from typing import Dict, List

def prepare_dataset_sample(
        sample: Dict,
        labelmap: Dict,
        nlp: spacy.Language
    ):
    mapper = SpacySequenceMapper(sample["text"], nlp=nlp)
    raw_annotations = LabelLoader.from_text_spans(sample["label"], mapper)

    prepared_sample = raw_annotations\
        .filterEntries(lambda e: e[2] in labelmap)\
        .renameLabels(labelmap)\
        .withoutOverlaps("prefer_longest")\
        .toEntities(with_text=True)

    return prepared_sample

def evaluateDatasets(
        ground_truth: List[Dict],
        ground_truth_labelmap: Dict,
        prediction: List[Dict],
        prediction_labelmap: Dict,
        nlp: spacy.Language = None
    ):
    if nlp is None:
        nlp = spacy.blank("de")

    assert isinstance(nlp, spacy.Language)

    common_labels = set(
        ground_truth_labelmap.values())\
        .intersection(
            set(prediction_labelmap.values()
        )
    )

    # Filter label map to only contain common label classes
    ground_truth_labelmap = { k:v for k,v in ground_truth_labelmap.items() if v in common_labels }
    prediction_labelmap = { k:v for k,v in prediction_labelmap.items() if v in common_labels }

    assert common_labels, "No common labels found!"

    prep_ground_truth = [ prepare_dataset_sample(s, ground_truth_labelmap, nlp) for s in ground_truth ]
    prep_prediction = [ prepare_dataset_sample(s, prediction_labelmap, nlp) for s in prediction ]

    score_response = evaluate(
        prep_ground_truth,
        prep_prediction,
        span_match="overlap",
        type_match="exact"
    )

    return score_response["scores"]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_dataset", type=str, help="Path to evaluation dataset as JSONL file")
    parser.add_argument("evaluation_labelmap", type=str, help="Label class mapping or the evaluation dataset - as JSON")
    parser.add_argument("prediction_dataset", type=str, help="Path to predicted dataset as JSONL file")
    parser.add_argument("prediction_labelmap", type=str, help="Label class mapping or the prediction dataset - as JSON")
    parser.add_argument("-l", "--spacy_language", type=str, default="de", help="Spacy's language-dependent tokenizer")
    args = parser.parse_args()

    # Try to use GPU
    if torch.cuda.is_available():
        spacy.require_gpu(torch.cuda.current_device())

    # Load datasets
    with open(args.evaluation_dataset, "r") as f:
        evaluation = [ json.loads(l) for l in f.readlines() if l ]
    evaluation_labelmap = json.loads(args.evaluation_labelmap)

    with open(args.prediction_dataset, "r") as f:
        prediction = [ json.loads(l) for l in f.readlines() if l ]
    prediction_labelmap = json.loads(args.prediction_labelmap)

    # Run evaluation
    results = evaluateDatasets(
        evaluation,
        evaluation_labelmap,
        prediction,
        prediction_labelmap,
        spacy.blank(args.spacy_language)
    )

    print(json.dumps(results, indent=4))
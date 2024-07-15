#!/usr/bin/env python3
import os, sys
import subprocess as sp
from tempfile import TemporaryDirectory
from typing import List, Dict
import shutil, json, re, datetime

import spacy
from spacy.util import filter_spans
from spacy.tokens import DocBin
from spacy.language import Language
import torch

from sklearn.model_selection import train_test_split

def get_gpu_id() -> int:
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return -1

def decode_nlp(nlp) -> Language:
    # Enable GPU support before loading any model...
    if torch.cuda.is_available(): spacy.prefer_gpu()

    # Try to load nlp object
    if isinstance(nlp, Language):
        pass
    elif isinstance(nlp, str):
        try:
            # Try to load from path
            nlp = spacy.load(nlp)
        except OSError:
            try:
                # Try to load from blank
                nlp = spacy.blank(nlp)
            except ImportError:
                raise ValueError("Unknown nlp description: {}".format(nlp))
    else:
        raise ValueError("Unknown nlp object given.")

    return nlp

def replace_line_after_match(filename, ptns, new_line, append=False):
    """
    Use this method to easiy replace certain lines in the spacy config file.
    """
    with open(filename, "r") as f:
        text = f.read()

    lines = text.split("\n")

    line_cur = 0
    for ptn in ptns:
        if isinstance(ptn, re.Pattern):
            for l in lines[line_cur:]:
                if ptn.fullmatch(l):
                    break
                else: line_cur += 1
        elif isinstance(ptn, str):
            for l in lines[line_cur:]:
                if l.startswith(ptn):
                    break
                else: line_cur += 1
    if line_cur == len(lines):
        # do nothing
        pass
    else:
        if isinstance(new_line , str):
            # just replace line
            if append:
                lines = lines[:line_cur+1] + [new_line] + lines[lines_cur+1:]
            else:
                lines[line_cur] = new_line
        else:
            # add block of lines
            if append:
                lines[:line_cur+1] + new_line + lines[line_cur+1:]
            else:
                lines[:line_cur] + new_line + lines[line_cur+1:]

        with open(filename, "w") as f:
            f.write("\n".join(lines))

def samples_to_docbin(samples: List[Dict], nlp: Language) -> DocBin:
    """
    Convert JSONL-like samples ({"text": ..., "label": [ ... ] }) into SpaCy's DocBin
    """
    docs = DocBin()
    for sample in samples:
        text = sample["text"]
        doc = nlp(text)

        spans = []
        for label in sample["label"]:
            span = doc.char_span(label[0], label[1], label=label[2], alignment_mode="expand")
            if span is not None:
                spans.append(span)
        try:
            doc.ents = spans
        except ValueError:
            print("Overlapping spans were detected. Removing shorter span...", file=sys.stderr)
            doc.ents = filter_spans(spans)
        docs.add(doc)
    return docs

def make_spacy_config(
        base_dir: str,
        model_name: str,
        language: str,
        learning_rate: float = 5e-5,
        batch_size: int = 128,
        epochs: int = 10,
        seed: int = 0
    ):
    spacy_cfg = os.path.join(base_dir, "base.cfg")

    # Make base config
    sp.check_call([
        "python3", "-m", "spacy", "init", "config",
        spacy_cfg,
        "-l", language,
        "-p", "ner",
        "-G",
        "-o", "accuracy"
    ])
    replace_line_after_match(spacy_cfg, ["[components.transformer.model]", "name"], 'name = "{}"'.format(model_name))
    replace_line_after_match(spacy_cfg, ["[training.optimizer.learn_rate]", "initial_rate"], "initial_rate = {}".format(learning_rate))
    replace_line_after_match(spacy_cfg, ["[nlp]", "batch_size"], "batch_size = {}".format(batch_size))
    replace_line_after_match(spacy_cfg, ["[training]", "seed ="], "seed = {}".format(seed))
    replace_line_after_match(spacy_cfg, ["[training]", "max_epochs ="], "max_epochs = {}".format(epochs))
    replace_line_after_match(spacy_cfg, ["[training]", "max_steps ="], "max_steps = 0")

    spacy_cfg_final = os.path.join(base_dir, "final.cfg")
    sp.check_call(["python3", "-m", "spacy", "init", "fill-config", spacy_cfg, spacy_cfg_final])

def make_spacy_train(base_dir):
    cfg_path = os.path.join(base_dir, "final.cfg")
    corpus_train = os.path.join(base_dir, "train.spacy")
    corpus_dev = os.path.join(base_dir, "dev.spacy")
    output_path = os.path.join(base_dir, "output")
    gpu_id = get_gpu_id()

    # Write start time:
    with open(os.path.join(base_dir, "time_start.txt"), "w") as f:
        f.write("Started training at: {}\n".format(
            datetime.datetime.now().strftime('%a %d %b %Y, %I:%M%p'))
        )

    # Run train log
    train_result = sp.run([
        "python3", "-m", "spacy", "train",
        cfg_path,
        "--paths.train", corpus_train,
        "--paths.dev", corpus_dev,
        "--output", output_path,
        "--gpu-id", str(gpu_id),
        "-V"
    ], stdout=sp.PIPE, stderr=sp.STDOUT, check=True)

    # Write train log
    with open(os.path.join(base_dir, "train_log.txt"), "wb") as f:
        f.write(train_result.stdout)

    # Write stop time:
    with open(os.path.join(base_dir, "time_stop.txt"), "w") as f:
        f.write("Stopped training at: {}\n".format(
            datetime.datetime.now().strftime('%a %d %b %Y, %I:%M%p'))
        )

def ner_train(
        train_samples: List[Dict],
        dev_samples: List[Dict],
        nlp: Language,
        output_dir: str,
        model_path: str = "uklfr/gottbert-base",
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 5e-5
    ):

    if os.path.exists(output_dir):
        raise ValueError("Directory already exists: {}".format(output_dir))

    nlp = decode_nlp(nlp)

    with TemporaryDirectory() as tdir:
        # Load & save samples
        train_docs = samples_to_docbin(train_samples, nlp).to_disk(os.path.join(tdir, "train.spacy"))
        dev_docs = samples_to_docbin(dev_samples, nlp).to_disk(os.path.join(tdir, "dev.spacy"))

        # Prepare config
        make_spacy_config(
            tdir,
            model_path,
            nlp.lang,
            learning_rate = learning_rate,
            batch_size = batch_size,
            epochs = epochs,
        )

        # Run training
        make_spacy_train(tdir)

        shutil.copytree(tdir, output_dir)

def ner_train_single(
        samples: List[Dict],
        nlp: Language,
        output_dir: str,
        model_path: str = "uklfr/gottbert-base",
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 5e-5,
        split_traineval_test: float = 0.1,
        split_train_eval: float = 0.2
    ):
    # Break samples into: <-- train / dev --><-- test -->
    traindev_samples, test_samples = train_test_split(samples, test_size=split_traineval_test, shuffle=False)
    # Break train / dev samples into: <-- train --><-- dev -->
    train_samples, dev_samples = train_test_split(traindev_samples, test_size=split_train_eval, shuffle=False)

    ner_train(
        train_samples,
        dev_samples,
        nlp,
        output_dir,
        model_path = model_path,
        epochs = epochs,
        batch_size = batch_size,
        learning_rate = learning_rate,
    )

def ner_predict(text: str, nlp: Language) -> List[Dict]:
    # Load model
    nlp = decode_nlp(nlp)

    # Return entities
    return [ (ent.start_char, ent.end_char, ent.label_) for ent in nlp(text).ents ]

def ner_predict_single(
        samples: List[Dict],
        nlp: Language
    ) -> List[Dict]:

    # Try to enable GPU before loading the model
    if torch.cuda.is_available(): spacy.prefer_gpu()

    # Manually try to decipher the potential model paths
    if isinstance(nlp, str) and os.path.exists(os.path.join(nlp, "meta.json")):
        # nlp seems to point to a SpaCy model directory
        nlp = spacy.load(nlp)
    elif isinstance(nlp, str) and os.path.exists(os.path.join(nlp, "output", "model-best", "meta.json")):
        # nlp seems to point to the base directory of a SpaCy train session
        nlp = spacy.load(os.path.join(nlp, "output", "model-best"))
    elif isinstance(nlp, str) and os.path.exists(os.path.join(nlp, "spacy", "output", "model-best", "meta.json")):
        # nlp seems to point to the base directory of a generic session
        nlp = spacy.load(os.path.join(nlp, "spacy", "output", "model-best"))

    nlp = decode_nlp(nlp)

    samples_output = []
    for sample in samples:
        samples_output.append({
            "text": sample["text"],
            "label": ner_predict(sample["text"], nlp)
        })

    return samples_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        parser.add_argument("cmd", help="command", choices=["train"], type=str)
        parser.add_argument("jsonl_file", help="Path to jsonl corpus file", type=str)
        parser.add_argument("output_dir", help="Path to output dir (Must not exist!)", type=str)
        parser.add_argument("-m", "--model_path", help="Huggingface model path", default="uklfr/gottbert-base", type=str)
        parser.add_argument("-l", "--nlp", help="SpaCy language path", default="de", type=str)
        parser.add_argument("-n", "--batch_size", help="Batch size", default=128, type=int)
        parser.add_argument("-e", "--epochs", help="Epochs", default=10, type=int)
        parser.add_argument("-r", "--learning_rate", help="Learning rate", default=5e-5, type=float)
        parser.add_argument("-s0", "--split_test", help="Split of test set vs. train+dev set", default=0.1, type=float)
        parser.add_argument("-s1", "--split_dev", help="Split of dev set vs. train set", default=0.2, type=float)

        args = parser.parse_args()

        # Load dataset
        with open(args.jsonl_file, "r") as f:
            samples = [ json.loads(l) for l in f.readlines() if l ]

        ner_train_single(
            samples,
            args.nlp,
            args.output_dir,
            model_path = args.model_path,
            epochs = args.epochs,
            batch_size = args.batch_size,
            learning_rate = args.learning_rate,
            split_traineval_test = args.split_test,
            split_train_eval = args.split_dev
        )

    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        parser.add_argument("cmd", help="command", choices=["predict"], type=str)
        parser.add_argument("jsonl_input_file", help="Path to jsonl input file", type=str)
        parser.add_argument("jsonl_output_file", help="Path to jsonl output file", type=str)
        parser.add_argument("spacy_model", help="Path to SpaCy model", type=str)

        args = parser.parse_args()

        # Load dataset
        with open(args.jsonl_input_file, "r") as f:
            samples_input = [ json.loads(l) for l in f.readlines() if l ]

        # Predict
        samples_output = ner_predict_single(
            samples_input,
            args.spacy_model
        )

        # Write output
        with open(args.jsonl_output_file, "w") as f:
            f.write("\n".join([ json.dumps(s) for s in samples_output ]))

    else:
        print("Commands 'train' or 'predict' are expected.", file=sys.stderr)
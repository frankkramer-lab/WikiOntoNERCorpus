"""Data Loader for WikiNER weakly annotated dataset"""

import json, sys
from datasets import Dataset, Features, Sequence, Value, ClassLabel, Split, Array2D
from tqdm import tqdm
from textwrap import dedent
from functools import reduce
import pandas as pd
from typing import Tuple, Literal, Union

def loadDataset(
        wiki_jsonl_file: str,
        tokenizer,
        max_length: int = 512,
        text_key: str = "text",
        label_key: str = "label",
        neg_label_key: str = "neg_label",
        unknown_loss_scaling: float = 0.1,
        label_loss_scaling: Union[Tuple[float, float],Literal["balanced"]] = "balanced",
        return_as_dataframe: bool = False
    ):
    _CITATION = dedent("""\
    TBA
    """)

    _DESCRIPTION = dedent("""\
    WikiOntoNER weakly annotated dataset
    """)

    # Load items
    with open(wiki_jsonl_file, "r") as f:
        docs = [ json.loads(l) for l in tqdm(f.readlines(), desc="Loading JSONL lines") if l ]

    n_docs = len(docs)
    print(f"Found {n_docs} lines", file=sys.stderr)

    # Parse items
    docs = [
        {
            "text": doc[text_key],
            "label": doc[label_key],
            "neg": doc[neg_label_key]
        }
        for doc in tqdm(docs, desc="Extracting items", total=n_docs)
    ]

    # Extracting base labels
    base_labels = [
        set([ lbl[2] for lbl in doc["label"] ])
        for doc in tqdm(docs, desc="Collecting label classes", total=n_docs)
    ]
    base_labels = reduce(lambda x,y: x.union(y), base_labels, set())
    print(f"Found base labels: {base_labels}", file=sys.stderr)

    iob2_labels = ["O"]
    for lbl in base_labels:
        iob2_labels += [f"B-{lbl}", f"I-{lbl}"]

    id2label = { i_lbl:lbl for i_lbl, lbl in enumerate(iob2_labels) }
    label2id = { lbl:i_lbl for i_lbl, lbl in enumerate(iob2_labels) }
    patched_label2id = { "neg": -1, **label2id}
    print(f"Using the following iob2 labels: {iob2_labels}", file=sys.stderr)

    # Apply tokenization with NER tagging sequence and loss scaling
    minimal_docs = []
    n_positive_label_tokens = 0
    n_negative_label_tokens = 0
    for doc in tqdm(docs, desc="Tokenization and NER chunk sequence loading", total=n_docs):
        # Tokenize first
        tokenized_text = tokenizer(
            doc["text"],
            return_offsets_mapping=True,
            truncation=True,
            padding='max_length',
            max_length=max_length
        )

        # Prepare label sequence and loss scaling sequence
        token_sequence = [patched_label2id["O"]] * len(tokenized_text['input_ids'])

        # Extracting actual labels
        for label in doc["label"]:
            # Skip empty labels
            if label[0] >= label[1]: continue

            t_start_idx = tokenized_text.char_to_token(label[0])
            # Get the last token that is *inside* the span
            t_stop_idx = tokenized_text.char_to_token(label[1] - 1)

            if t_start_idx is not None and t_stop_idx is not None:
                t_stop_idx += 1 # Make up for the exclusive stop idx
                # Ignore zero-length labels
                if t_start_idx == t_stop_idx: continue

                # Assign label to tokens within the span
                token_sequence[t_start_idx] = patched_label2id["B-" + label[2]]
                for t_in_idx in range(t_start_idx + 1, t_stop_idx):
                    token_sequence[t_in_idx] = patched_label2id['I-' + label[2]]

                # Update statistics
                n_positive_label_tokens += t_stop_idx - t_start_idx

        # Extracting negative labels
        for neg_label in doc["neg"]:
            # Skip empty labels
            if neg_label[0] >= neg_label[1]: continue

            t_start_idx = tokenized_text.char_to_token(neg_label[0])
            # Get the last token that is *inside* the span
            t_stop_idx = tokenized_text.char_to_token(neg_label[1] - 1)

            if t_start_idx is not None and t_stop_idx is not None:
                t_stop_idx += 1 # Make up for the exclusive stop idx
                # Ignore zero-length labels
                if t_start_idx == t_stop_idx: continue

                for t_in_idx in range(t_start_idx, t_stop_idx):
                    token_sequence[t_in_idx] = patched_label2id["neg"]

                # Update statistics
                n_negative_label_tokens += t_stop_idx - t_start_idx

        # Truncate or pad token labels to max_length
        if len(token_sequence) > max_length:
            token_sequence = token_sequence[:max_length]
        else:
            n_tokens_to_fill = len(token_sequence)
            token_sequence += [label2id["O"]] * (max_length - n_tokens_to_fill)

        minimal_docs.append({
            "text": doc["text"],
            "input_ids": tokenized_text["input_ids"],
            "attention_mask": tokenized_text["attention_mask"],
            "labels": token_sequence,
            "offsets": tokenized_text["offset_mapping"]
        })

    # Now, add loss scaling info
    if label_loss_scaling == "balanced" and n_negative_label_tokens == 0 or n_positive_label_tokens == 0:
        print(
            "Cannot balance labels since #positive or #negative labels is zero.\n"\
            "Continue with loss scaling 1.0/1.0",
            file=sys.stderr
        )
        label_loss_scaling = (1.0, 1.0)
    elif label_loss_scaling == "balanced":
        label_loss_scaling = (
            n_negative_label_tokens / n_positive_label_tokens, # Scaling for positive labels
            n_positive_label_tokens / n_negative_label_tokens  # Scaling for negative labels
        )
    print(f"Using loss scaling factors: {label_loss_scaling[0]:.3f} (pos), {label_loss_scaling[1]:.3f} (neg), {unknown_loss_scaling:.3f} (unknown)", file=sys.stderr)

    # Load loss scaling
    docs_with_loss = []
    for i_doc, doc in tqdm(enumerate(minimal_docs), desc="Add loss scaling", total=n_docs):
        loss_scaling = [
            unknown_loss_scaling if token_class == 0 else \
            label_loss_scaling[1] if token_class == -1 else \
            label_loss_scaling[0]
            for token_class in doc["labels"]
        ]
        # Also replace 'fake' neg label class
        labels = [ 0 if token_class == -1 else token_class for token_class in doc["labels"]]
        docs_with_loss.append({
            **doc,
            "loss_scaling": loss_scaling,
            "labels": labels,
        })

    # Load into DataFrame
    df = pd.DataFrame(data=docs_with_loss, columns=[
        "input_ids",
        "attention_mask",
        "text",
        "loss_scaling",
        "labels",
        "offsets"
    ])
    if return_as_dataframe:
        return df, (label2id, id2label)
    else:
        print("Parsing DataFrame into HF Dataset schema...", file=sys.stderr)
        ds = Dataset.from_pandas(
            df,
            features = Features({
                "input_ids": Sequence(Value('int32')),
                "attention_mask": Sequence(Value('int32')),
                "text": Value("string"),
                "labels": Sequence(
                    ClassLabel(
                        num_classes = len(label2id),
                        names = [ lbl for i, lbl in sorted(id2label.items(), key=lambda x:x[0]) ]
                    )
                ),
                "loss_scaling": Sequence(Value('float32')),
                "offsets": Array2D(shape=(-1, 2), dtype='int32')
            }),
            split = Split.ALL
        )
        return ds, (label2id, id2label)

if __name__ == "__main__":
    # Example Usage
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    model_path="uklfr/gottbert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ds, (lbl2id, id2lbl) = loadDataset("dataset.jsonl", tokenizer)
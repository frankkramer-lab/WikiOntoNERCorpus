#!/usr/bin/env python3
import os, sys, json
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import RobertaForTokenClassification, RobertaTokenizer
import numpy as np
from wikionto_dataset import loadDataset
from wikionto_utils import loadConfig
from tqdm import tqdm

dpath = os.path.dirname(os.path.abspath(__file__))

def applyWeaklyDataImputation(
        tokenizer_path: str,
        model_path: str,
        weakly_dataset_path: str,
        applied_dataset_path: str
    ):
    # Load the model and tokenizer
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load dataset
    dataset_all, (label2id, id2label) = loadDataset(weakly_dataset_path, tokenizer)

    with torch.no_grad():
        model = RobertaForTokenClassification.from_pretrained(model_path,
            id2label=id2label,
            label2id=label2id
        ).to(device)

        # Determine all positions of I-<LabelClass> to mask
        basic_inside_mask = np.array([ float("-inf") if lbl.startswith("I-") else 0.0 for lbl, _ in sorted(label2id.items(), key=lambda x: x[1])])
        def infmask_except(idx):
            m = basic_inside_mask.copy()
            m[idx] = 0.0
            return m

        id2insideid = { idx:label2id["I-" + lbl[2:]] for idx, lbl in id2label.items() if lbl != "O" }
        id2insidemask = {
            idx:(
                basic_inside_mask if lbl == "O" else\
                infmask_except(id2insideid[idx])
            )
            for idx, lbl in id2label.items()
        }

        samples_corrected = []
        for sample in tqdm(dataset_all, desc="Inferring samples from weakly annotated dataset", total=len(dataset_all)):
            pred = model(
                input_ids=torch.tensor(sample["input_ids"]).reshape(1,-1).to(device),
                attention_mask=torch.tensor(sample["attention_mask"]).reshape(1,-1).to(device)
            )

            logits = pred.logits[0,:].cpu().numpy()

            # decode IOB2 NER states...
            ner_sequence = []
            previous_class_id = label2id["O"]
            for token_idx, token_logits in enumerate(logits):
                # Check for EOS token
                if sample["input_ids"][token_idx] == tokenizer.eos_token_id:
                    # Stop if we reached the end of the sequence eariler
                    break

                # Check for non-str token
                if sample["offsets"][token_idx][0] == sample["offsets"][token_idx][1]:
                    # This token yields an empty span
                    ner_sequence.append(label2id["O"])
                    previous_class_id = label2id["O"]
                    continue

                # Mask by adding -inf to masked values
                masked_token_logits = token_logits + id2insidemask[previous_class_id]
                # get greedy best next guess
                next_class_id = masked_token_logits.argmax()

                ner_sequence.append(next_class_id)
                previous_class_id = next_class_id

            error = None
            try:
                labels = []
                label_buffer = None
                for token_idx, tok_lbl_id in enumerate(ner_sequence):
                    lbl = id2label[tok_lbl_id]

                    # Check for EOS token
                    if sample["input_ids"][token_idx] == tokenizer.eos_token_id:
                        # Stop if we reached the end of the sequence eariler
                        break

                    if label_buffer is not None and (lbl == "O" or lbl.startswith("B-")):
                        # flush label buffer
                        lbl_class, token_positions = label_buffer
                        begin_pos = min([ sample["offsets"][tpos][0] for tpos in token_positions ])
                        end_pos = max([ sample["offsets"][tpos][1] for tpos in token_positions ])
                        labels.append( (begin_pos, end_pos, lbl_class) )
                        label_buffer = None

                    if lbl.startswith("B-"):
                        label_buffer = (lbl[2:], [token_idx])
                    elif lbl.startswith("I-"):
                        assert label_buffer is not None
                        assert lbl[2:] == label_buffer[0]
                        label_buffer[1].append(token_idx)

                # flush buffer at the end as well
                if label_buffer is not None:
                    # flush label buffer
                    lbl_class, token_positions = label_buffer
                    begin_pos = min([ sample["offsets"][tpos][0] for tpos in token_positions ])
                    end_pos = max([ sample["offsets"][tpos][1] for tpos in token_positions ])
                    labels.append( (begin_pos, end_pos, lbl_class) )
                    label_buffer = None

            except Exception as e:
                error = e

                import code
                import readline
                import rlcompleter
                vars = globals()
                vars.update(locals())
                readline.set_completer(rlcompleter.Completer(vars).complete)
                readline.parse_and_bind("tab: complete")
                code.InteractiveConsole(vars).interact()

            samples_corrected.append({
                "text": sample["text"],
                "label": labels
            })

        print(f"Writing corrected dataset to file: {applied_dataset_path}", file=sys.stderr)
        with open(applied_dataset_path, "w") as f:
            f.write("\n".join([ json.dumps(sample) for sample in samples_corrected ]))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="Path to cfg file or raw input (json)", type=str)
    args = parser.parse_args()

    cfg = loadConfig(args.cfg)
    best_checkpoint_path = f"{cfg['session_name']}/results/checkpoint-best"
    dataset_path = f"{cfg['session_name']}/dataset.jsonl"
    dataset_fixed = f"{cfg['session_name']}/dataset_applied.jsonl"

    applyWeaklyDataImputation(
        best_checkpoint_path, # Tokenizer path (is also stored after training)
        best_checkpoint_path, # Model path
        dataset_path, # Dataset (input)
        dataset_fixed # Dataset (output)
    )
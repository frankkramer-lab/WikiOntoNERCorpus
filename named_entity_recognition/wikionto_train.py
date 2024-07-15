#!/usr/bin/env python3
import os, sys, json
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import RobertaForTokenClassification, RobertaTokenizer
from transformers import Trainer, TrainingArguments
import numpy as np
from functools import reduce
import evaluate
from wikionto_dataset import loadDataset
from wikionto_utils import loadConfig
from typing import Tuple, Union, Literal

dpath = os.path.dirname(os.path.abspath(__file__))

def runTraining(
        dataset_file: str,
        session_name: str = "default",
        output_folder: str = None,
        model_path: str = "uklfr/gottbert-base",
        learning_rate: float = 5e-5,
        batch_size: int = 32,
        epochs: int = 3,
        split_traineval_test: float = 0.1,
        split_train_eval: float = 0.2,
        unknown_loss_scaling: float = 0.1,
        label_loss_scaling: Union[Tuple[float, float],Literal["balanced"]] = "balanced",
        *args,
        **kwargs
    ):
    if args or kwargs:
        print(f"Unknown arguments: {repr({**{ a:None for a in args}, **kwargs})}", file=sys.stderr)

    if output_folder is None:
        output_folder = dpath
    output_path = os.path.join(output_folder, session_name)

    checkpoint_path = os.path.join(output_path, "results")
    logging_path = os.path.join(output_path, "logs")

    # Load the model and tokenizer
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load dataset
    dataset_all, (label2id, id2label) = loadDataset(
        dataset_file,
        tokenizer,
        unknown_loss_scaling=unknown_loss_scaling,
        label_loss_scaling=label_loss_scaling
    )

    # Split dataset
    ds_dict = dataset_all.train_test_split(split_traineval_test)
    ds_traineval = ds_dict["train"].train_test_split(split_train_eval)

    ds_test = ds_dict["test"]
    ds_train = ds_traineval["train"]
    ds_eval = ds_traineval["test"]

    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=logging_path,
        logging_strategy='epoch',
        remove_unused_columns=False,
        learning_rate=learning_rate,
        metric_for_best_model="loss"
    )
    print(f"Training Arguments:\n{json.dumps(training_args.to_dict(), indent=2)}\n", file=sys.stderr)

    model = RobertaForTokenClassification.from_pretrained(model_path,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # Customized Trainer for handling scalar multiplication
    class ScalarTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            scalars = inputs.pop("loss_scaling")
            offsets = inputs.pop("offsets")

            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            loss = loss * scalars.view(-1)  # Multiply loss by scalars
            return (loss.mean(), outputs) if return_outputs else loss.mean()

    def compute_custom_metrics(p):
        metric1 = evaluate.load("precision")
        metric2 = evaluate.load("recall")
        metric3 = None # evaluate.load("f1")
        metric4 = evaluate.load("accuracy")

        logits, labels = p
        predictions = np.argmax(logits, axis=-1)
        mode = "binary" if len(id2label) == 2 else "macro"
        if mode == "binary":
            metric3 = evaluate.load("f1", average=mode)
        else:
            metric3 = evaluate.load("f1", average=mode)

        flat_preds = predictions.flatten()
        flat_lbls = labels.flatten()

        precision = metric1.compute(predictions=flat_preds, references=flat_lbls, average=mode)["precision"]
        recall = metric2.compute(predictions=flat_preds, references=flat_lbls, average=mode)["recall"]
        f1 = metric3.compute(predictions=flat_preds, references=flat_lbls, average=mode)["f1"]
        accuracy = metric4.compute(predictions=flat_preds, references=flat_lbls)["accuracy"]

        # apply seqeval as well
        str_predictions = np.vectorize(lambda x: id2label[x])(predictions)
        str_labels = np.vectorize(lambda x: id2label[x])(labels)
        seqeval_metric = evaluate.load('seqeval')


        per_label_info = reduce(lambda x,y: {**x, **y}, [
            {k:v} if not isinstance(v, dict) else\
            { f"{k}_{subk}": subv for subk, subv in v.items() }
            for k,v in seqeval_metric.compute(predictions=str_predictions, references=str_labels).items()
        ], {})

        return {"raw_precision": precision, "raw_recall": recall, "raw_f1": f1, "raw_accuracy": accuracy, **per_label_info}

    # Create trainer
    trainer = ScalarTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        compute_metrics=compute_custom_metrics
    )

    # Training
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model(os.path.join(checkpoint_path, "checkpoint-best"))
    tokenizer.save_pretrained(os.path.join(checkpoint_path, "checkpoint-best"))
    metrics["train_samples"] = len(ds_train)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(ds_eval)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    print("Finished training.", file=sys.stderr)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="Path to cfg file or raw input (json)", type=str)
    args = parser.parse_args()

    cfg = loadConfig(args.cfg)
    runTraining(**cfg)
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:13:12 2023

@author: zilio
"""

import argparse
import sys
from datasets import load_dataset

import evaluate
import logging
import numpy as np

import transformers
print(transformers.__version__)
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import MBart50TokenizerFast


def arguments():
    parser = argparse.ArgumentParser(description="Fine-tune NMT models")
    parser.add_argument("-m", "--model", default="facebook/mbart-large-50-many-to-many-mmt", help="Folder path: path to model that is going to be fine-tuned.")
    parser.add_argument("-d", "--dataset", default="training_data/", help="Folder path: path to fine-tuning dataset -- must contain a train and dev TSV file.")
    parser.add_argument("-dn", "--dev_name", default="dev.tsv", help="Name of the dev file.")
    parser.add_argument("-tn", "--train_name", default="train.tsv", help="Name of the train file.")
    return parser.parse_args()

args = arguments()

################################
# Setting up logging info
logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
logger.setLevel(logging.INFO)

transformers.logging.set_verbosity_info() # Should provide log info to the console
################################

model_checkpoint = args.model

data_files = {"train": args.train_name, "validation": args.dev_name}
raw_datasets = load_dataset(args.dataset, data_files=data_files)
metric = evaluate.load("sacrebleu")

print("Data splits:\n", raw_datasets)
print("First instance of the training set:\n", raw_datasets["train"][0])
print("This is the metric:\n", metric)


print("Loading tokenizer...")
src_lang = "pt_XX"
tokenizer = MBart50TokenizerFast.from_pretrained(model_checkpoint, src_lang=src_lang)
print("Tokenizer loaded!")

###########################################
# Defining parameters for the tokenization
prefix = ""
max_input_length = 512
max_target_length = 512
source_lang = "por"
target_lang = "pob"
model_name = model_checkpoint.split("/")[-1]
model_dir = f"{model_name}-finetuned-{source_lang}-to-{target_lang}"

# This function needs to be modified according to the format of the loaded dataset
def preprocess_function(examples):
    inputs = examples[source_lang] #This should be a list of strings (each string is a non-tokenized sentence/segment)
    targets = examples[target_lang] #This should be a list of strings (each string is a non-tokenized sentence/segment)
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    print("Tokenized inputs")
    # Setup the tokenizer for targets
    references = tokenizer(targets, max_length=max_target_length, truncation=True)
    print("References are tokenized")
    model_inputs["labels"] = references["input_ids"]
    return model_inputs

print("Tokenizing dataset...")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
print("Tokenized!")

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

############################################
# Defining args for fine-tuning
print("Setting up fine-tuning arguments...")
batch_size = 4
args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    save_strategy="epoch",
    num_train_epochs=100,
    predict_with_generate=True,
    report_to="none",
    logging_steps = 100,
    metric_for_best_model="bleu",
    load_best_model_at_end=True
)
print("Parameters are set!")
############################################

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#####################################################
# Defining functions for the evaluation step
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
#####################################################


# Settig up the trainer for the fine-tuning task
print("Setting up trainer...")
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
print("Trainer is set!")

# Fine-tuning
print("Fine-tuning started...")
trainer.train()
print("Finished fine-tuning!")
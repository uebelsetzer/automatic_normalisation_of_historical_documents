# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:02:48 2023

@author: zilio
"""

import argparse
import evaluate

###########################################################
def arguments():
    parser = argparse.ArgumentParser(description="Evaluate translations using SacreBLEU")
    parser.add_argument("-s", "--source", default="test_sets/medical_test.txt", help="File containing source sentences.")
    parser.add_argument("-n", "--normalisation", default="normalised/medical_opus.txt", help="File containing one automatic normalisation per line.")
    parser.add_argument("-g", "--gold", default="gold/medical_gold.txt", help="File containing one reference translation per line.")
    parser.add_argument("-r", "--results", default="results/results_medical_opus_ft.txt", help="Output file for the evaluation.")
    return parser.parse_args()

def get_sentences(f):
    sentences = list()
    for line in f:
        line = line.strip()
        sentences.append(line)
    return sentences

##########################################################

args = arguments()
sacrebleu = evaluate.load("sacrebleu")

source_file = args.source
predictions_file = args.normalisation
reference_file = args.gold
results_file = args.results

with open(source_file, "r", encoding="utf-8") as src:
    source = get_sentences(src)

with open(predictions_file, "r", encoding="utf-8") as pred:
    predictions = get_sentences(pred)

with open(reference_file, "r", encoding="utf-8") as ref:
    references = [[x] for x in get_sentences(ref)]

# Computes the overall BLEU score for all sentences
results = sacrebleu.compute(predictions=predictions, references=references)
print(results)

# Computes the BLEU score for each line
with open(results_file, "w", encoding="utf-8") as res:
    res.write("source\tgold\ttranslation\tBLEU score\n")
    for s, p, r in zip(source, predictions, references):
        # Added use_effective_order for sentence-by-sentence evaluation, 
        # so sentences smaller than 4 tokens are also evaluated
        line_results = sacrebleu.compute(predictions=[p], references=r, use_effective_order=True)
        res.write(f"{s}\t{r[0]}\t{p}\t{line_results['score']}\n")
    
    res.write(f"\n\nFinal score:\t{results}")
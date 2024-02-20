# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:03:26 2023

@author: zilio
"""

import argparse
import datetime
from bs4 import BeautifulSoup as bs
from string import punctuation
from nltk import word_tokenize

##########################################################
def arguments():
    parser = argparse.ArgumentParser(description="Modernises a text using a glossary")
    parser.add_argument("-m", "--mode", choices=["txt", "tmx"], default="txt", help="Type of input file. The options available are txt or tmx.")
    parser.add_argument("-f", "--file", default="test_sets/medical_test.txt", help="File containing one sentence per line.")
    parser.add_argument("-t", "--tmx", default="test_sets/medical_test.tmx", help="TMX file.")
    parser.add_argument("-g", "--glossary", default="glossary.txt", help="A standard OmegaT glossary file in a two-column TSV format.")
    parser.add_argument("-o", "--output", default="test/ps_translation_rule", help="File containing one translation per line.")
    parser.add_argument("-tmx", "--to-tmx", default=True, help="Binary value indicating whether a TMX file should be generated.")
    parser.add_argument("-e", "--to-eval", default=True, help="Binary value indicating whether an evaluation file should be generated.")
    return parser.parse_args()


def process_tmx(src):
    src_text = list()
    
    soup = bs(''.join(src), "lxml")
    
    for tu in soup.find_all("tu"):
        segs = [x.text for x in tu.find_all("seg")]
        assert len(segs) == 2, "TUs should have only two different segments: source and target."
        src_text.append(segs[0])
    
    return src_text


def process_txt(src):
    src_text = list()
    
    for line in src:
        line = line.strip()
        if line != "":
            line = line.split("\t")
            src_text.append(line[0])
    
    return src_text


def generate_tmx(src_list, translations, filename):
    assert len(src_list) == len(translations), f"The source and target have different sizes: {len(src_list)} vs. {len(translations)}\nPlease make sure they have the same amount of sentences."
    with open(filename, "w", encoding="utf-8") as tmx_file:
        tmx_file.write("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE tmx SYSTEM "tmx14.dtd">
<tmx version="1.4">
  <header creationtool="Python Script" o-tmf="OmegaT TMX" adminlang="EN-US" datatype="plaintext" creationtoolversion="6.0.0_0_1bf1729c" segtype="sentence" srclang="pt"/>
  <body>
""")
        for src, tgt in zip(src_list, translations):
            now = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")
            print(f"SOURCE: {src}\nTARGET: {tgt}\n{now}\n\n")
            tmx_file.write(f"""    <tu>
      <tuv xml:lang="pt">
        <seg>{src}</seg>
      </tuv>
      <tuv xml:lang="pt-BR" creationid="translate.py" creationdate="{now}">
        <seg>{tgt}</seg>
      </tuv>
    </tu>\n""")
    
        tmx_file.write("""  </body>\n</tmx>""")

def generate_eval(translations, filename):
    with open(filename, "w", encoding="utf-8") as eval_file:
        for target in translations:
            eval_file.write(target + "\n")

##########################################################

args = arguments()

mode = args.mode
filename = args.tmx if mode == "tmx" else args.file
glossary = args.glossary
output_file = args.output
to_tmx = args.to_tmx
to_eval = args.to_eval

translation_dict = dict()
bi_grams_dict = dict()

print("Extracting glossary...")
g = open(glossary, "r", encoding="utf-8")

for line in g:
    line = line.strip()
    if line != "":
        line = line.split("\t")
        if " " in line[0]:
            bi_grams_dict[line[0]] = line[1]
        else:
            translation_dict[line[0]] = line[1]

g.close()
print("Glossary is ready!")


print("Processing source text...")
f = open(filename, "r", encoding="utf-8")

source_text = f.readlines()

f.close()

if mode == "tmx":
    src_text = process_tmx(source_text)
else:
    src_text = process_txt(source_text)
print("Source text processed!")

print("Normalising sentences...")
normalised_text = list()

for st in src_text:
    normalised_sentence = str()
    is_open_punct = False
    
    st = word_tokenize(st, language="portuguese")
    for t in st:
        if t == "&" or t not in punctuation:
            if t in translation_dict:
                t = translation_dict[t]
            elif t[:-1] in translation_dict:
                t = translation_dict[t[:-1]] + t[-1]
            
            if is_open_punct:
                normalised_sentence = normalised_sentence + t
                is_open_punct = False
            else:
                normalised_sentence = normalised_sentence + f" {t}"
        else:
            if t in "[(":
                is_open_punct = True
                normalised_sentence = normalised_sentence + f" {t}"
            else:
                normalised_sentence = normalised_sentence + t
    
    for bi in bi_grams_dict:
        if bi in normalised_sentence:
            normalised_sentence.replace(bi, bi_grams_dict[bi])
    
    normalised_text.append(normalised_sentence.strip())
    
if to_tmx:
    tmx_file = output_file + ".tmx"
    generate_tmx(src_text, normalised_text, tmx_file)
else:
    for nt in normalised_text:
        print(nt)

if to_eval:
    eval_file = output_file + ".txt"
    generate_eval(normalised_text, eval_file)

print("Finished!")



                
# Automatic normalisation for historical documents

This repository contains data associated to the paper:


> Zilio, Leonardo, Lazzari, Rafaela R. and Finatto, Maria José B. 2024. Can rules still beat neural networks? The case of automatic normalisation for 18th-century Portuguese texts. In _Proceedings of the International Conference on Computational Processing of Portuguese (PROPOR 2024): workshops_, vol. 2, pp. xxx-xxx. (to appear)


The repository contains the datasets for training and evaluating neural machine translation (NMT) models on historical data, as well as a rule-based counterpart. There are three main file types in the root folder of this repository: fine-tuning scripts for NMT, translating scripts (for applying the fine-tuned NMT models and the rule-based model to a test set), and an evaluation script. These are all individually explained below.

The rest of the data is divided into several folders, containing the training set `training_data/`, the test set `test_sets/`, the gold standard (i.e. manually normalised data) for each test set `gold/`, the automatically normalised data `normalised/`, which is available in both TXT and TMX formats, and the results for all the experiments `results/`.


## Basic requirements for running the scripts in this repository

A basic requirement is to have **Python 3** installed in your system. It is also recommended to create a new virtual environment for running the scripts in this repository.



### On Linux:

A simple way to do create a new environment would be to open a command line (or terminal) within the folder with this repository and type:

```
python -m venv nmt4hd
source nmt4hd/bin/activate
```

### On Windows:

A simple way to do create a new environment would be to open a PowerShell window (make sure you run it as administrator) within the folder with this repository and type:

```
pip install virtualenv
python -m virtualenv nmt4dh
.\nmt4dh\Scripts\activate
```


This should create and activate a new environment called **nmt4hd** (this name can be modified), and make sure it will not mess with anything else in the root Python environment.

With the activated virtual environment a few required packages need to be installed. The full list is in the "requirements.txt" file. All packages can be installed at once from within this repository folder by simply typing:

```
pip install -r requirements.txt
```

After the installation is done, new models can be fine-tuned using the commands in the following section.


## Fine-tuning neural machine translation (NMT) models

Due to the size of the fine-tuned models, we could not host them in this repository, but all scripts are provided to reproduce the work conducted on teh paper.

There are three scripts for fine-tuning NMT models, whose names start with "finetuningNMT-". These files can be run as they are, without selecting any option, by simply typing:
```
python finetuningNMT-opus.py
```

or
```
python finetuningNMT-mbart50.py
```

or
```
python finetuningNMT-nllb_200.py
```

or these scripts can be customised (for using other available models, or a different training set) by adding some options. If they are not customised, they will run with the default options used for one of the experiments described in the paper. The available options are the following (which are equal for all models):

- -m = path to the model 
- -d = path to dataset folder 
- -dn = name of the dev file 
- -tn = name of the train file


Example (which is in the default use of the script):
```
python finetuningNMT-opus.py -m "Helsinki-NLP/opus-mt-tc-big-itc-itc" -d "training_data/" -dn "dev.tsv" -tn "train.tsv"
```

For fine-tuning NMT models, we strongly recommend using a good GPU (for the experiments described in the paper, we used an Nvidia RTX 4090 24GB). Another option would be to use online services, such as Google Colab to run the scripts on the virtual machines. The fine-tuning process could potentially be run on CPU, especially for the OPUS model, which is smaller, but it would be extremely slow.


## Normalising the test set with baseline/fine-tuned models

After a model is fine-tuned, the best model will be saved to a folder that will look like this:

"opus-mt-tc-big-itc-itc-finetuned-por-to-pob/"

Inside this folder, there will be a few more folders that look like this:

"checkpoint-297/"

At the end of the fine-tuning step, the system will save the best model evaluated during the training and it will inform which checkpoint it was, so make sure to take note of that number as it will be in the name of the folder that should be loaded during the normalisation step.

Again there are three scripts for normalising texts (all starting with "translate-"). They all can be used with their default parameters, as in the fine-tuning step, but at least the folder of the fine-tuned model must be informed, for instance:

```
python finetuningNMT-opus.py -m "opus-mt-tc-big-itc-itc-finetuned-por-to-pob/checkpoint-297/"
```

It is also possible to load an already existing model, for instance: "Helsinki-NLP/opus-mt-tc-big-itc-itc", which will then translate/normalise the source text using the non-fine-tuned model.

Further parameters/options that can be changed (for all models):

- -s = path to the source text (i.e. in our case the test set)
- -o = path for the output file (i.e. the file with normalised output)
- -tmx = indicate to the script that a TMX file should be generated at the end. Valid options are True or False (default=True).
- -e = indicate to the script that an evaluation file should be generated at the end (this file is used in the next step for evaluating the results). Valid options are True or False (default=True).

This parameter/option only exists for OPUS models:
- -l = indicate the language that the model should translate into. The format is a 3-character code, for instance, ">>pob<<"". This option does not work well with fine-tuned models, so the default is set to "".


## Normalising texts with the rule-based script

The rule-based method uses a glossary `glossary.txt` as input, and normalises a text based on matching entries. It also tokenises the text prior to the matching process, using NLTK's word tokeniser.

To run the rule-based script, simply type:

```
python rule-based_normalisation.py
```

The medical test set will be used by default, and it will generate an evaluation TXT file, and a bilingual (i.e. original -> normalised) TMX file.

Options that can be changed here are:

- -m = the format of the input file. Valid options are "txt" or "tmx" (default="txt").
- -f = path to input TXT file (if using "txt" mode above)
- -t = path to input TMX file (if using "tmx" mode above)
- -g = path to glossary file
- -o = output file name
- -tmx = indicate to the script that a TMX file should be generated at the end. Valid options are True or False (default=True).
- -e = indicate to the script that an evaluation file should be generated at the end (this file is used in the next step for evaluating the results). Valid options are True or False (default=True).


## Evaluating the results

There is only one script for evaluating all models (including the rule-based model, despite the name of the script): `evaluate_NMT.py`

This script uses a source text, an automatically normalised output and a gold standard (a normalisation made by humans) as input, and produces a TSV file with an overall BLEU score for the whole set and an individual BLEU score for each sentence. This TSV file can also be used for error analysis, as it contains all three stages of the sentences (original, automatically normalised, and normalised by human).

If used as is, it will take as input an already existing normalisation produced with the OPUS fine-tuned model for the medical domain, as presented in the paper:

```
python evaluate_NMT.py
```

The options for this script are:

- -s = path to the source text with historical spelling. This is used only for generating the output with the three stages of the text. It is not used for computing BLEU scores
- -n = path to the automatically normalised text
- -g = path to the gold standard file
- -r = name of the results file that is going to be generated
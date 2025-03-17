# Named Entity Recognition (NER) Project

This repository contains implementation of Named Entity Recognition (NER) systems for both English and Tagalog languages using BERT-based models.

## Project Overview

Named Entity Recognition (NER) is a subtask of information extraction that seeks to locate and classify named entities in text into predefined categories such as person names, organizations, locations, etc. This project implements NER systems using fine-tuned BERT models.

## Repository Structure

- [`bert-base-uncased-for-NER.ipynb`](bert-base-uncased-for-NER.ipynb): Jupyter notebook for training a BERT-based NER model
- [`english_ner.py`](english_ner.py): Python script for English NER
- [`pretrained_english_ner.py`](pretrained_english_ner.py): Python script using pretrained English NER model
- [`tagalog_ner.py`](tagalog_ner.py): Python script for Tagalog NER
- [`english_ner_model`](english_ner_model): Directory containing the pretrained English NER model files

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install required packages
pip install transformers torch numpy pandas
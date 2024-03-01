import math
import sys
from collections import defaultdict
import os
from nltk.corpus import stopwords
import string
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from scipy.spatial.distance import cosine
import torch.nn.functional as F

def compress_sentence(sentences): # compress sentence
    # Load model 
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("../data/sentence-compression")
    model = AutoModelForTokenClassification.from_pretrained("../data/sentence-compression")
    compress = pipeline("token-classification", model=model, tokenizer=tokenizer)
    compressed_sentences = []
    for sentence in sentences:
        compressed_sentence = []
        labels = compress(sentence)
        #import pdb; pdb.set_trace()
        for label in labels:
            if label['entity'] == 'LABEL_1':
                compressed_sentence.append(label['word'])
        compressed_sentence = ' '.join(compressed_sentence)
        compressed_sentences.append(compressed_sentence)
    return compressed_sentences


if __name__ == "__main__":
    input_file = sys.argv[1]
    with open(input_file, 'r') as f:
        sentences = f.readlines()
        compressed_sentences = compress_sentence(sentences)
    
    for sentence in compressed_sentences:
        print(sentence)
    
    
    
        
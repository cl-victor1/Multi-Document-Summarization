'''
helper functions:
'''

import re; import nltk;  nltk.download('stopwords');  from nltk.corpus import stopwords
from collections import Counter; from typing import List, Dict; import string; import math; import numpy as np;
# nltk.download('punkt');  nltk.download('averaged_perceptron_tagger');

import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def is_contentful_sentence(sentence_as_string):
    doc = nlp(sentence_as_string)
    has_subject = any(token.dep_.endswith("subj") for token in doc)
    has_verb = any(token.pos_ == "VERB" for token in doc)
    contains_entity = len(doc.ents) > 0

    # grammatical = "grammatically complete" if has_subject and has_verb else "grammatically incomplete"
    # entity_presence = "contains" if contains_entity else "does not contain"
    # print(f"Sentence is {grammatical}, and {entity_presence} a named entity.")

    return has_subject and has_verb and contains_entity

# Example sentence
test_sentence = "Apple is looking at buying U.K. startup for $1 billion."
is_contentful_sentence(test_sentence)


# word processing
def process_words(sentence_as_string, ngram, remove_stop_words = remove_stopwords_global):

    # must contain: subject, verb, named entity
    if not is_contentful_sentence(sentence_as_string):
        return [];

    list_of_words = sentence_as_string.split();
    result  = None;

    punctuation_pattern = f'^[{re.escape(string.punctuation)}]+$'
    # punctuation_re = re.compile(r'[^\w\s]|_')

    list_of_words = [word for word in list_of_words if (not bool(re.match(punctuation_pattern, word))) and (not "'s" in word)]

    if remove_stop_words:
        list_of_words = [word for word in list_of_words[1:-1] if word.lower() not in stop_words];
    else:
        pass;

    list_of_words.insert(0, '<s>');
    list_of_words.append('</s>');

    if ngram == 2:
        result = [(list_of_words[i], list_of_words[i + 1]) for i in range(len(list_of_words) - 1)]
    else:
        result = list_of_words[1:-1]


    if len(result) < length_cut:
        result = [];
    return result
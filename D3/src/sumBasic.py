

## assume the following structure:

"""
all_files
  ----- devtest_output
          ----- docA1
          ----- docA2
          ...          
  ----- training_output
          ----- docA1
          ----- docA2
          ...          
  ----- evaltest_output
          ----- docA1
          ----- docA2
          ...          
"""

import sys;
args = sys.argv;
path_to_your_files = args[1]; # e.g:
                              # LING-575-project/D2/outputs

import re

#######################################################################
## Define a function for extracting HEADLINE, DATELINE, TEXT as a DICT
#######################################################################

def get_TEXT(file_path):
    """
    Clean the text file by correctly identifying HEADLINE, DATELINE, and TEXT.
    Improve sentence splitting in the TEXT section based on the assumption that a sentence
    ends with a punctuation followed by a newline.

    :param file_path: Path to the file that needs to be cleaned
    :return: Dictionary with cleaned HEADLINE, DATELINE, and TEXT sections
    """
    # Initialize a dictionary to store the information
    cleaned_info = {
        "HEADLINE": "",
        "DATELINE": "",
        "TEXT": []
    }

    # Flags to indicate the current section being read
    is_headline = False
    is_dateline = False
    is_text = False

    # Regex pattern to identify punctuation (excluding periods and hyphens)
    # punctuation_re = re.compile(r'[^\w\s\-]')
    punctuation_re = re.compile(r'[.!?\']')

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Remove any leading/trailing whitespace and \n's
            stripped_line = line.strip()
            # print(stripped_line)

            # Identify the section based on flags and content
            if stripped_line.startswith('HEADLINE:'):
                is_headline = True
                cleaned_info["HEADLINE"] += stripped_line[len('HEADLINE:'):].strip() + " "
            elif stripped_line.startswith('DATELINE:'):
                is_headline = False
                is_dateline = True
                cleaned_info["DATELINE"] += stripped_line[len('DATELINE:'):].strip() + " "
            elif stripped_line == "" and is_dateline:
                is_dateline = False
                is_text = True
            elif is_headline:
                cleaned_info["HEADLINE"] += stripped_line + " "
            elif is_dateline:
                cleaned_info["DATELINE"] += stripped_line + " "
            elif is_text:
                # Improve sentence splitting: Sentence ends with a punctuation followed by \n
                if stripped_line == "":
                    continue;
                else:
                    if cleaned_info["TEXT"]:
                        if punctuation_re.search(cleaned_info["TEXT"][-1]):  # Check if the last line ends with a specified punctuation
                            cleaned_info["TEXT"].append(stripped_line);
                        else:
                            cleaned_info["TEXT"][-1] += ' ' + stripped_line;
                    else:
                        cleaned_info["TEXT"].append(stripped_line);

    # Remove any trailing whitespace from HEADLINE and DATELINE
    cleaned_info["HEADLINE"] = cleaned_info["HEADLINE"].strip()
    cleaned_info["DATELINE"] = cleaned_info["DATELINE"].strip()

    return cleaned_info


##################################################
## Get Text cleaned, and Get Word Probability
##################################################

import os
from collections import Counter
from typing import List, Dict
import string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# word processing
def process_words(sent_as_string, ngram = 1, remove_stop_words = True):

    list_of_words = sent_as_string.split();
    result  = None;

    punctuation_pattern = f'^[{re.escape(string.punctuation)}]+$'
    # punctuation_re = re.compile(r'[^\w\s]|_')

    list_of_words = [word for word in list_of_words if not bool(re.match(punctuation_pattern, word))]

    list_of_words.insert(0, '<s>');
    list_of_words.append('</s>');

    if ngram == 1:
        if remove_stop_words:
            result = [word for word in list_of_words[1:-1] if word.lower() not in stop_words];
        else:
            result = list_of_words[1:-1]
    elif ngram == 2:
        result = [(list_of_words[i], list_of_words[i + 1]) for i in range(len(list_of_words) - 1)]

    if len(result) < 5:
        result = [];
    return result

# Define the function to process all files in a directory
def get_structured_data_from_path_v0(directory: str, ngram = 1):
    training_dir = directory+"/training_output";

    training_original_backup_by_file = {}; # texts by file before cleaning

    training_1gram_v0  = {};               # texts by file after cleaning

    training_original_backup_by_doc = {};  # texts aggregated by doc before cleaning

    training_1gram_v1  = {};               # texts aggregated by doc after cleaning


    # Walk through the directory
    for docSetA in os.listdir(training_dir):
        docSetA_dir = training_dir+"/"+docSetA;

        training_original_backup_by_file[docSetA]={};
        training_1gram_v0[docSetA]={};

        training_original_backup_by_doc[docSetA]=[];
        training_1gram_v1[docSetA]=[];


        for file in os.listdir(docSetA_dir):
            file_path = os.path.join(docSetA_dir, file)

            training_1gram_v0[docSetA][file] = [];
            training_original_backup_by_file[docSetA][file] = [];

            sentences_list = get_TEXT(file_path)['TEXT'];

            for sent in sentences_list:
                training_original_backup_by_file[docSetA][file].append(sent);
                training_original_backup_by_doc[docSetA].append(sent);
                words = process_words(sent, ngram);

                # if len <= 4, ignore the sentence
                # if words == []:
                #     continue;
                # else: store data

                training_1gram_v0[docSetA][file].append(words);
                training_1gram_v1[docSetA].append(words);

    return {0: training_1gram_v0, 1: training_1gram_v1, 2: training_original_backup_by_file, 3: training_original_backup_by_doc};

prelim_unigram_data_v0 = get_structured_data_from_path_v0(path_to_your_files, ngram = 1);

# Define the function to calculate word probabilities
def get_word_probabilities(data = prelim_unigram_data_v0[0], ngram = 1):

    word_counter = Counter();

    # Walk through the directory
    for docSetA in data:
        for file in data[docSetA]:
            sentences_list = data[docSetA][file];
            for sent_as_words_list in sentences_list:
                word_counter.update(sent_as_words_list);

    total_words = sum(word_counter.values())
    word_probabilities = {word: count / total_words for word, count in word_counter.items()}

    result = {0: data, 1: word_counter, 2: word_probabilities};

    return result;

unigram_inter_results = get_word_probabilities();
unigram_inter_results_agg = {k:v for k,v in unigram_inter_results.items()};
unigram_inter_results_agg[0] = prelim_unigram_data_v0[1]

##############################################
## Get Preliminary (1st round) sentence scores
################################################

# Define the function to calculate the sentence score
def calculate_sentence_score(sentence: List[str], word_probabilities: Dict[str, float], power = 1) -> float:
    if len(sentence) == 0:
        return 0;
    if power == 1:
        result = sum(word_probabilities.get(word, 0) for word in sentence) / len(sentence);
    elif power == 2:
        result = sum((word_probabilities.get(word, 0))**2 for word in sentence) / len(sentence)
    return result

# Define the function to process all files in a directory
def get_preliminary_sentence_scores(data = unigram_inter_results[0], word_probabilities = unigram_inter_results[2]):

    # Dictionary to hold the sentence scores for each file
    training_sentence_scores = {};

    # Walk through the directory by file
    for docSetA in data:
        training_sentence_scores[docSetA] = {};
        for file in data[docSetA]:
            training_sentence_scores[docSetA][file] = [];

            for sent in data[docSetA][file]:
                # Calculate and store the sentence scores
                sentence_scores = calculate_sentence_score(sent, word_probabilities)
                training_sentence_scores[docSetA][file].append(sentence_scores)

    return training_sentence_scores

# preliminary sentence scores
unigram_scores_preliminary = get_preliminary_sentence_scores()
unigram_scores_preliminary_agg = {};
# aggregate by document
for docSetA in unigram_scores_preliminary:
    unigram_scores_preliminary_agg[docSetA] = [];
    for file in unigram_scores_preliminary[docSetA]:
        for sent in unigram_scores_preliminary[docSetA][file]:
            unigram_scores_preliminary_agg[docSetA].append(sent);
            
            


##############################################################
## SUMBASIC by file
##############################################################

import numpy as np;

# Define the function to collect enough sentences to form the summary.

def get_final_sentence_scores(data = unigram_inter_results[0], word_probabilities = unigram_inter_results[2], scores = unigram_scores_preliminary, perc = 0.2, n_words = 100, n_sents = 2):

    # Dictionary to hold the sentence scores for each file
    training_summary_perc = {};
    training_summary_n_sents = {};
    training_summary_n_words = {};

    # Walk through the directory
    for docSetA in data:
        training_summary_perc[docSetA] = {};
        training_summary_n_sents[docSetA] = {};
        training_summary_n_words[docSetA] = {};

        for file in data[docSetA]:
            training_summary_perc[docSetA][file] = None;
            training_summary_n_sents[docSetA][file] = None
            training_summary_n_words[docSetA][file] = None

            if data[docSetA][file] == []:
                continue;

            # file_length defined as number of words in this doc file
            file_length = sum([len(sent) for sent in data[docSetA][file]]);


            # result for training_summary_perc
            # initialize
            current_file_scores = [sent_score for sent_score in scores[docSetA][file]]; # this is a list of scores of each sentence
            current_top_idx = [np.argmax(current_file_scores)];
            current_cumulative_len = len(data[docSetA][file][current_top_idx[-1]]);
            while current_cumulative_len/file_length < perc:
                # update the score of the top
                current_file_scores[current_top_idx[-1]] = calculate_sentence_score(data[docSetA][file][current_top_idx[-1]], word_probabilities, power = 2);
                # update the cumulative length
                current_cumulative_len += len(data[docSetA][file][np.argmax(current_file_scores)])
                # update the top idx list
                current_top_idx.append(np.argmax(current_file_scores))
            # training_summary_perc[docSetA][file] = [" ".join(data[docSetA][file][idx]) for idx in current_top_idx];
            training_summary_perc[docSetA][file] = [idx for idx in current_top_idx];

            # result for training_summary_n_sent
            # re-initialize
            current_file_scores = [sent_score for sent_score in scores[docSetA][file]]; # this is a list of scores of each sentence
            current_top_idx = [np.argmax(current_file_scores)];
            current_cumulative_len = 1;
            if len(scores[docSetA][file]) <= n_sents:
                training_summary_n_sents[docSetA][file] = data[docSetA][file];
            else:
                while current_cumulative_len < n_sents:
                    # update the score of the top
                    current_file_scores[current_top_idx[-1]] = calculate_sentence_score(data[docSetA][file][current_top_idx[-1]], word_probabilities, power = 2);
                    # update the cumulative length
                    current_cumulative_len += 1
                    # update the top idx list
                    current_top_idx.append(np.argmax(current_file_scores))
            # training_summary_n_sents[docSetA][file] = [" ".join(data[docSetA][file][idx]) for idx in current_top_idx]
            training_summary_n_sents[docSetA][file] = [idx for idx in current_top_idx]

            # result for training_summary_n_words
            # re-initialize
            current_file_scores = [sent_score for sent_score in scores[docSetA][file]]; # this is a list of scores of each sentence
            current_top_idx = [np.argmax(current_file_scores)];
            current_cumulative_len = len(data[docSetA][file][current_top_idx[-1]]);
            if file_length <= n_words:
                training_summary_n_words[docSetA][file] = data[docSetA][file];
            else:
                while current_cumulative_len < n_words:
                    # update the score of the top
                    current_file_scores[current_top_idx[-1]] = calculate_sentence_score(data[docSetA][file][current_top_idx[-1]], word_probabilities, power = 2);
                    # update the cumulative length
                    current_cumulative_len += len(data[docSetA][file][np.argmax(current_file_scores)])
                    # update the top idx list
                    current_top_idx.append(np.argmax(current_file_scores))
            # training_summary_n_words[docSetA][file] = [" ".join(data[docSetA][file][idx]) for idx in current_top_idx];
            training_summary_n_words[docSetA][file] = [idx for idx in current_top_idx];


    return {'perc': training_summary_perc, 'n_sents': training_summary_n_sents, "n_words": training_summary_n_words}

# preliminary sentence scores
sumBasic_idx_by_file = get_final_sentence_scores()

def get_sumBasic_by_file(arg = sumBasic_idx_by_file):
    summary_perc_cleaned = {};
    summary_n_sents_cleaned = {};
    summary_n_words_cleaned = {};

    summary_perc = {};
    summary_n_sents = {};
    summary_n_words = {};

    for docSetA in arg['perc']:
        summary_perc_cleaned[docSetA] = {}
        summary_perc[docSetA] = {}
        for file in arg['perc'][docSetA]:
            if arg['perc'][docSetA][file] == None:
                continue;
            summary_perc_cleaned[docSetA][file] = [" ".join(prelim_unigram_data_v0[0][docSetA][file][idx]) for idx in arg['perc'][docSetA][file]]
            summary_perc[docSetA][file] = [prelim_unigram_data_v0[2][docSetA][file][idx] for idx in arg['perc'][docSetA][file]]

    for docSetA in arg['n_sents']:
        summary_n_sents_cleaned[docSetA] = {}
        summary_n_sents[docSetA] = {}
        for file in arg['n_sents'][docSetA]:
            if arg['n_sents'][docSetA][file] == None:
                continue;
            summary_n_sents_cleaned[docSetA][file] = [" ".join(prelim_unigram_data_v0[0][docSetA][file][idx]) for idx in arg['n_sents'][docSetA][file]]
            summary_n_sents[docSetA][file] = [prelim_unigram_data_v0[2][docSetA][file][idx] for idx in arg['n_sents'][docSetA][file]]

    for docSetA in arg['n_words']:
        summary_n_words_cleaned[docSetA] = {};
        summary_n_words[docSetA] = {};
        for file in arg['n_words'][docSetA]:
            if arg['n_words'][docSetA][file] == None:
                continue;
            summary_n_words_cleaned[docSetA][file] = [" ".join(prelim_unigram_data_v0[0][docSetA][file][idx]) for idx in arg['n_words'][docSetA][file]]
            summary_n_words[docSetA][file] = [prelim_unigram_data_v0[2][docSetA][file][idx] for idx in arg['n_words'][docSetA][file]]

    return {"perc": {0: summary_perc_cleaned, 1:summary_perc}, "n_sents": {0: summary_n_sents_cleaned,1:summary_n_sents}, "n_words": {0: summary_n_words_cleaned,1:summary_n_words}}

sumBasic = get_sumBasic_by_file()

##############################################################
## SUMBASIC by doc
##############################################################

import numpy as np;

# Define the function to collect enough sentences to form the summary.

def get_final_sentence_scores_agg(data = unigram_inter_results_agg[0], word_probabilities = unigram_inter_results_agg[2], scores = unigram_scores_preliminary_agg, perc = 0.1, n_words = 100, n_sents = 2):

    # Dictionary to hold the sentence scores for each file
    training_summary_perc = {};
    training_summary_n_sents = {};
    training_summary_n_words = {};

    # Walk through the directory
    for docSetA in data:
        training_summary_perc[docSetA] = [];
        training_summary_n_sents[docSetA] = [];
        training_summary_n_words[docSetA] = [];

        if data[docSetA] == []:
            continue;

        # file_length defined as number of SENTENCES in this doc file
        doc_length_sentences = len(scores[docSetA]);
        file_length_words = sum([len(sent) for sent in data[docSetA]]);

        # result for training_summary_perc
        # initialize
        current_doc_scores = [sent_score for sent_score in scores[docSetA]]; # this is a list of scores of each sentence
        current_top_idx = [np.argmax(current_doc_scores)];
        current_cumulative_len = 1;
        if len(scores[docSetA]) <= n_sents:
            training_summary_n_sents[docSetA] = data[docSetA];
        else:
            while current_cumulative_len/doc_length_sentences < perc:
                # update the score of the top
                current_doc_scores[current_top_idx[-1]] = calculate_sentence_score(data[docSetA][current_top_idx[-1]], word_probabilities, power = 2);
                # update the cumulative length
                current_cumulative_len += 1
                # update the top idx list
                current_top_idx.append(np.argmax(current_doc_scores))
        # training_summary_perc[docSetA] = [" ".join(data[docSetA][idx]) for idx in current_top_idx]
        training_summary_perc[docSetA] = [idx for idx in current_top_idx]

        # result for training_summary_n_sent, training_summary_perc
        # initialize
        current_doc_scores = [sent_score for sent_score in scores[docSetA]]; # this is a list of scores of each sentence
        current_top_idx = [np.argmax(current_doc_scores)];
        current_cumulative_len = 1;
        if len(scores[docSetA]) <= n_sents:
            training_summary_n_sents[docSetA] = data[docSetA];
        else:
            while current_cumulative_len < n_sents:
                # update the score of the top
                current_doc_scores[current_top_idx[-1]] = calculate_sentence_score(data[docSetA][current_top_idx[-1]], word_probabilities, power = 2);
                # update the cumulative length
                current_cumulative_len += 1
                # update the top idx list
                current_top_idx.append(np.argmax(current_doc_scores))
        # training_summary_n_sents[docSetA] = [" ".join(data[docSetA][idx]) for idx in current_top_idx]
        training_summary_n_sents[docSetA] = [idx for idx in current_top_idx]

        # result for training_summary_n_words
        # re-initialize
        current_doc_scores = [sent_score for sent_score in scores[docSetA]]; # this is a list of scores of each sentence
        current_top_idx = [np.argmax(current_doc_scores)];
        current_cumulative_len = len(data[docSetA][current_top_idx[-1]]);
        if file_length_words <= n_words:
            training_summary_n_words[docSetA] = data[docSetA];
        else:
            while current_cumulative_len < n_words:
                # update the score of the top
                current_doc_scores[current_top_idx[-1]] = calculate_sentence_score(data[docSetA][current_top_idx[-1]], word_probabilities, power = 2);
                # update the cumulative length
                current_cumulative_len += len(data[docSetA][np.argmax(current_doc_scores)])
                # update the top idx list
                current_top_idx.append(np.argmax(current_doc_scores))
        # training_summary_n_words[docSetA] = [" ".join(data[docSetA][idx]) for idx in current_top_idx];
        training_summary_n_words[docSetA] = [idx for idx in current_top_idx];


    return {'perc': training_summary_perc, 'n_sents': training_summary_n_sents, "n_words": training_summary_n_words}

# preliminary sentence scores
sumBasic_idx_by_doc = get_final_sentence_scores_agg();


def get_sumBasic_by_doc(arg = sumBasic_idx_by_doc):
    summary_perc_cleaned = {};
    summary_n_sents_cleaned = {};
    summary_n_words_cleaned = {};

    summary_perc = {};
    summary_n_sents = {};
    summary_n_words = {};

    for docSetA in arg['perc']:
        summary_perc_cleaned[docSetA] = []
        summary_perc[docSetA] = []
        if not arg['perc'][docSetA]:
            continue;
        summary_perc_cleaned[docSetA] = [" ".join(prelim_unigram_data_v0[1][docSetA][idx]) for idx in arg['perc'][docSetA]]
        summary_perc[docSetA] = [prelim_unigram_data_v0[3][docSetA][idx] for idx in arg['perc'][docSetA]]

    for docSetA in arg['n_sents']:
        summary_n_sents_cleaned[docSetA] = []
        summary_n_sents[docSetA] = []

        if not arg['n_sents'][docSetA]:
            continue;
        summary_n_sents_cleaned[docSetA] = [" ".join(prelim_unigram_data_v0[1][docSetA][idx]) for idx in arg['n_sents'][docSetA]]
        summary_n_sents[docSetA]         = [prelim_unigram_data_v0[3][docSetA][idx] for idx in arg['n_sents'][docSetA]]

    for docSetA in arg['n_words']:
        summary_n_words_cleaned[docSetA] = {};
        summary_n_words[docSetA] = {};
        if not arg['n_words'][docSetA]:
            continue;
        summary_n_words_cleaned[docSetA] = [" ".join(prelim_unigram_data_v0[1][docSetA][idx]) for idx in arg['n_words'][docSetA]]
        summary_n_words[docSetA]         = [prelim_unigram_data_v0[3][docSetA][idx] for idx in arg['n_words'][docSetA]]

    return {"perc": {0: summary_perc_cleaned, 1:summary_perc}, "n_sents": {0: summary_n_sents_cleaned,1:summary_n_sents}, "n_words": {0: summary_n_words_cleaned,1:summary_n_words}}

# preliminary sentence scores
sumBasic_agg = get_sumBasic_by_doc()


#############################
## print result
#############################

def print_result():
    for docSetA in sumBasic_agg['n_words'][1]:
        print(f"{docSetA} - Summary : ")
        for i, sent in enumerate(sumBasic_agg['n_words'][1][docSetA]):
            print(f"{i+1}, {sent}")
        print()

print_result()

import os;
import sys;
args = sys.argv;
path_to_your_files = args[1]; # e.g:
                              # LING-575-project/D2/outputs
if args[2] == '-ry':
    remove_stopwords_gloabl = True;  ## -ry
else:
    remove_stopwords_gloabl = False; ## -rn

if args[3] == "-2":
    ngram_global = 2; "-2"
else:
    ngram_global = 1; "-1"

if len(args) >= 5:
    if args[4] == 'tfidf':
        word_score = 'tfidf'; 
    else:
        word_score = "prob";
else:
    word_score = 'prob';

import re;

def get_TEXT(file_path):

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

    ah = "DATE_TIME:";
    # ah = "DATELINE:";

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
            elif stripped_line.startswith(ah):
                is_headline = False
                is_dateline = True
                cleaned_info["DATELINE"] += stripped_line[len(ah):].strip() + " "
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
                            if cleaned_info["TEXT"][-1].startswith('(') and cleaned_info["TEXT"][-1].endswith(')'):
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


from collections import Counter
from typing import List, Dict
import string
import math

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set([sw.lower() for sw in stopwords.words("english")])

# word processing
def process_words(sent_as_string, ngram, remove_stop_words = remove_stopwords_global):

    list_of_words = sent_as_string.split();
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

    if ngram == 1:
        result = list_of_words[1:-1]
    elif ngram == 2:
        result = [(list_of_words[i], list_of_words[i + 1]) for i in range(len(list_of_words) - 1)]

    if len(result) < 5:
        result = [];
    return result

# Define the function to process all files in a directory
def get_structured_data_from_path_v0(directory: str, ngram):
    # training_dir = directory+"/training_output";
    training_dir = directory+"/devtest_output";

    training_original_backup_by_file = {}; # texts by file before cleaning

    training_1gram_v0_by_file  = {};       # texts by file after cleaning

    training_original_backup_by_doc = {};  # texts aggregated by doc before cleaning

    training_1gram_v0_by_doc  = {};        # texts aggregated by doc after cleaning


    # Walk through the directory
    for docSetA in os.listdir(training_dir):
        docSetA_dir = training_dir+"/"+docSetA;

        training_original_backup_by_file[docSetA]={};
        training_1gram_v0_by_file[docSetA]={};

        training_original_backup_by_doc[docSetA]=[];
        training_1gram_v0_by_doc[docSetA]=[];


        for file in os.listdir(docSetA_dir):
            file_path = os.path.join(docSetA_dir, file)

            training_1gram_v0_by_file[docSetA][file] = [];
            training_original_backup_by_file[docSetA][file] = [];

            sentences_list = get_TEXT(file_path)['TEXT'];

            for sent in sentences_list:
                training_original_backup_by_file[docSetA][file].append(sent);
                training_original_backup_by_doc[docSetA].append(sent);
                words = process_words(sent, ngram);

                training_1gram_v0_by_file[docSetA][file].append(words);
                training_1gram_v0_by_doc[docSetA].append(words);

    return {0: training_1gram_v0_by_file, 1: training_1gram_v0_by_doc, 2: training_original_backup_by_file, 3: training_original_backup_by_doc};


## CHANGE ngram = 2 to get bigram
prelim_unigram_data_v0 = get_structured_data_from_path_v0(path_to_your_files, ngram = ngram_global);

# Define the function to calculate word probabilities
def get_word_probabilities(data = prelim_unigram_data_v0[0]):

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

unigram_inter_results_by_file = get_word_probabilities();
unigram_inter_results_by_doc = {k:v for k,v in unigram_inter_results_by_file.items()};
unigram_inter_results_by_doc[0] = prelim_unigram_data_v0[1]


################################################
## Get TFIDF
################################################

# https://courses.cs.washington.edu/courses/cse373/17au/project3/project3-2.html

import math

def get_tfidf_by_file(data):

    total_number_of_files = 0;
    for docSetA in data:
        for file in data[docSetA]:
            total_number_of_files += 1;

    tf_absolute = {};
    tf_relative = {}; # relative to the file, instead of relative to the document

    word_in_file = {}; # by all files

    for docSetA in data:

        tf_absolute[docSetA] = {};
        tf_relative[docSetA] = {};

        for file in data[docSetA]:
            tf_absolute[docSetA][file] = Counter();
            tf_relative[docSetA][file] = {};

            for sent in data[docSetA][file]:
                tf_absolute[docSetA][file].update(sent);

            for term in tf_absolute[docSetA][file]:
                tf_relative[docSetA][file][term] = tf_absolute[docSetA][file][term]/sum(tf_absolute[docSetA][file].values())
                if term not in word_in_file:
                    word_in_file[term] = 1;
                else:
                    word_in_file[term] += 1;

    tfidf = {};
    for docSetA in tf_relative:
        tfidf[docSetA] = {};
        for file in tf_relative[docSetA]:
            tfidf[docSetA][file] = {};
            for term in tf_relative[docSetA][file]:
                tfidf[docSetA][file][term] = tf_relative[docSetA][file][term] * math.log(total_number_of_files/(word_in_file[term]+1));
    return tfidf

tf_idf_result_by_file = get_tfidf_by_file(prelim_unigram_data_v0[0]);

def get_tfidf_by_doc(data):

    total_number_of_docs = 0;
    for docSetA in data:
        total_number_of_docs += 1;

    tf_absolute = {};
    tf_relative = {}; # relative to the doc

    word_in_docs = {}; # by all files

    for docSetA in data:

        tf_absolute[docSetA] = Counter();
        tf_relative[docSetA] = {};

        for sent in data[docSetA]:
            tf_absolute[docSetA].update(sent);

        for term in tf_absolute[docSetA]:
            tf_relative[docSetA][term] = tf_absolute[docSetA][term]/sum(tf_absolute[docSetA].values())
            if term not in word_in_docs:
                word_in_docs[term] = 1;
            else:
                word_in_docs[term] += 1;

    tfidf = {};
    for docSetA in tf_relative:
        tfidf[docSetA] = {};
        for term in tf_relative[docSetA]:
            tfidf[docSetA][term] = tf_relative[docSetA][term] * math.log(total_number_of_docs/(word_in_docs[term]+1));
    return tfidf

tf_idf_result_by_doc = get_tfidf_by_doc(prelim_unigram_data_v0[1]);

tf_idf_result = {'file': tf_idf_result_by_file, 'doc': tf_idf_result_by_doc}


####################################################
## Calculating sentence weights
####################################################

# Define the function to calculate the sentence score
def calculate_sentence_score_prob(sentence: List[str], word_probabilities: Dict[str, float], power = 1) -> float:
    if len(sentence) == 0:
        return 0;
    if power == 1:
        result = sum(word_probabilities.get(word, 0) for word in sentence) / len(sentence);
    elif power == 2:
        result = sum((word_probabilities.get(word, 0))**2 for word in sentence) / len(sentence)
    return result

def calculate_sentence_score_tfidf_by_file(sentence, tfidf_doc_file, power = 1) -> float:
    if len(sentence) == 0:
        return 0;
    if power == 1:
        result = sum(tfidf_doc_file.get(word, 0) for word in sentence) / len(sentence);
    elif power == 2:
        result = sum((tfidf_doc_file.get(word, 0))/4 for word in sentence) / len(sentence)
    return result

def calculate_sentence_score_tfidf_by_doc(sentence, tfidf_doc, power = 1) -> float:
    if len(sentence) == 0:
        return 0;
    if power == 1:
        result = sum(tfidf_doc.get(word, 0) for word in sentence) / len(sentence);
    elif power == 2:
        result = sum((tfidf_doc.get(word, 0))/4 for word in sentence) / len(sentence)
    return result



##############################################################
## SUMBASIC by file
##############################################################

import numpy as np;

# Define the function to collect enough sentences to form the summary.

def get_final_sentence_scores_by_file(data = unigram_inter_results_by_file[0], word_probabilities = unigram_inter_results_by_file[2],
                                      tfidf = tf_idf_result, n_words = 100):

    # Dictionary to hold the sentence scores for each file
    all_sentence_scores_prob = {};
    all_sentence_scores_tfidf_by_file = {};

    # Walk through the directory by file
    for docSetA in data:
        all_sentence_scores_prob[docSetA] = {};
        all_sentence_scores_tfidf_by_file[docSetA] = {};
        for file in data[docSetA]:
            all_sentence_scores_prob[docSetA][file] = [];
            all_sentence_scores_tfidf_by_file[docSetA][file] = [];
            for sent in data[docSetA][file]:
                # Calculate and store the sentence scores
                sentence_scores = calculate_sentence_score_prob(sent, word_probabilities);
                all_sentence_scores_prob[docSetA][file].append(sentence_scores);
                sentence_scores_tfidf = calculate_sentence_score_tfidf_by_file(sent, tfidf['file'][docSetA][file]);
                all_sentence_scores_tfidf_by_file[docSetA][file].append(sentence_scores_tfidf);

    scores = {'prob': all_sentence_scores_prob, 'tfidf': all_sentence_scores_tfidf_by_file};

    # Dictionary to hold the sentence scores for each file
    training_summary_n_words = {};

    # Walk through the directory
    for docSetA in data:
        training_summary_n_words[docSetA] = {};

        for file in data[docSetA]:
            training_summary_n_words[docSetA][file] = None

            if data[docSetA][file] == []:
                continue;

            # file_length defined as number of words in this doc file
            file_length = sum([len(sent) for sent in data[docSetA][file]]);

            # result for training_summary_n_words
            # initialize
            current_file_scores = [sent_score for sent_score in scores[word_score][docSetA][file]];
            current_top_idx = [np.argmax(current_file_scores)];
            current_cumulative_len = len(data[docSetA][file][current_top_idx[-1]]);
            if file_length <= n_words:
                training_summary_n_words[docSetA][file] = data[docSetA][file];
            else:
                while current_cumulative_len < n_words:
                    # update the score of the top
                    if word_score == "tfidf":
                        current_file_scores[current_top_idx[-1]] = calculate_sentence_score_tfidf_by_file(data[docSetA][file][current_top_idx[-1]], tfidf['file'][docSetA][file], power = 2);
                    else:
                        current_file_scores[current_top_idx[-1]] = calculate_sentence_score_prob(data[docSetA][file][current_top_idx[-1]], word_probabilities, power = 2);
                    # update the cumulative length
                    current_cumulative_len += len(data[docSetA][file][np.argmax(current_file_scores)])
                    # update the top idx list
                    current_top_idx.append(np.argmax(current_file_scores))
            training_summary_n_words[docSetA][file] = [idx for idx in current_top_idx];


    return {"n_words": training_summary_n_words}

# preliminary sentence scores
sumBasic_idx_by_file = get_final_sentence_scores_by_file();

def get_sumBasic_by_file(arg = sumBasic_idx_by_file, ngram = ngram_global):
    summary_n_words_cleaned = {};

    summary_n_words = {};

    for docSetA in arg['n_words']:
        summary_n_words_cleaned[docSetA] = {};
        summary_n_words[docSetA] = {};
        for file in arg['n_words'][docSetA]:
            if arg['n_words'][docSetA][file] == None:
                continue;
            summary_n_words_cleaned[docSetA][file] = [prelim_unigram_data_v0[0][docSetA][file][idx] for idx in arg['n_words'][docSetA][file]]
            summary_n_words[docSetA][file]         = [prelim_unigram_data_v0[2][docSetA][file][idx] for idx in arg['n_words'][docSetA][file]]

    return {"n_words": {0: summary_n_words_cleaned,1:summary_n_words}}

sumBasic_by_file = get_sumBasic_by_file()



##############################################################
## SUMBASIC by doc
##############################################################

import numpy as np;

# Define the function to collect enough sentences to form the summary.

def get_final_sentence_scores_by_doc(data = unigram_inter_results_by_doc[0], word_probabilities = unigram_inter_results_by_doc[2],
                                     tfidf = tf_idf_result, n_words = 100):

    # Dictionary to hold the sentence scores for each file
    all_sentence_scores_prob = {};
    all_sentence_scores_tfidf_by_doc = {};

    # Walk through the directory by file
    for docSetA in data:
        all_sentence_scores_prob[docSetA] = [];
        all_sentence_scores_tfidf_by_doc[docSetA] = [];
        for sent in data[docSetA]:
            # Calculate and store the sentence scores
            sentence_scores = calculate_sentence_score_prob(sent, word_probabilities);
            all_sentence_scores_prob[docSetA].append(sentence_scores);
            sentence_scores_tfidf = calculate_sentence_score_tfidf_by_doc(sent, tfidf['doc'][docSetA]);
            all_sentence_scores_tfidf_by_doc[docSetA].append(sentence_scores_tfidf);

    scores = {'prob': all_sentence_scores_prob, 'tfidf': all_sentence_scores_tfidf_by_doc};

    # Dictionary to hold the sentence scores for each file
    training_summary_n_words = {};

    # Walk through the directory
    for docSetA in data:
        training_summary_n_words[docSetA] = [];

        if data[docSetA] == []:
            continue;

        # file_length defined as number of SENTENCES in this doc file
        doc_length_sentences = len(scores[word_score][docSetA]);
        file_length_words = sum([len(sent) for sent in data[docSetA]]);

        # result for training_summary_n_words
        # initialize
        current_doc_scores = [sent_score for sent_score in scores[word_score][docSetA]]; # this is a list of scores of each sentence
        current_top_idx = [np.argmax(current_doc_scores)];
        current_cumulative_len = len(data[docSetA][current_top_idx[-1]]);
        if file_length_words <= n_words:
            training_summary_n_words[docSetA] = data[docSetA];
        else:
            while current_cumulative_len < n_words:
                # update the score of the top
                if word_score == 'tfidf':
                    current_doc_scores[current_top_idx[-1]] = calculate_sentence_score_tfidf_by_doc(data[docSetA][current_top_idx[-1]], tfidf['doc'][docSetA], power = 2);
                else:
                    current_doc_scores[current_top_idx[-1]] = calculate_sentence_score_prob(data[docSetA][current_top_idx[-1]], word_probabilities, power = 2);
                # update the cumulative length
                current_cumulative_len += len(data[docSetA][np.argmax(current_doc_scores)])
                # update the top idx list
                current_top_idx.append(np.argmax(current_doc_scores))
        training_summary_n_words[docSetA] = [idx for idx in current_top_idx];

    return {"n_words": training_summary_n_words}

# preliminary sentence scores
sumBasic_idx_by_doc = get_final_sentence_scores_by_doc();


def get_sumBasic_by_doc(arg = sumBasic_idx_by_doc, ngram = ngram_global):

    summary_n_words_cleaned = {};

    summary_n_words = {};

    for docSetA in arg['n_words']:
        summary_n_words_cleaned[docSetA] = {};
        summary_n_words[docSetA] = {};
        if not arg['n_words'][docSetA]:
            continue;

        summary_n_words_cleaned[docSetA] = [prelim_unigram_data_v0[1][docSetA][idx] for idx in arg['n_words'][docSetA]];
        summary_n_words[docSetA]         = [prelim_unigram_data_v0[3][docSetA][idx] for idx in arg['n_words'][docSetA]]

    return {"n_words": {0: summary_n_words_cleaned,1:summary_n_words}}

# preliminary sentence scores
sumBasic_by_doc = get_sumBasic_by_doc()


#############################
## print result
#############################

def view_result():
    for k,v in sumBasic_by_doc['n_words'][1].items():
        print(k);
        for i, s in enumerate(v):
            print(f"{i}: {s}")
        print();

view_result()

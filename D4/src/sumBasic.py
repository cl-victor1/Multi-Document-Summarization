## python sumbasic.py input_path output_path arg3 arg4 arg5 arg6

args = sys.argv;

if args[3] == 'ry':
    remove_stopwords_global = True;            # arg3: '-ry' or '-rn'
else:
    remove_stopwords_global = False;
ngram_global = int(args[4][-1]);               # arg4: "-1" or "-2"
word_score = args[5];                          # arg5: 'tfidf' or "prob"
length_cut = int(args[6]);                     # arg6:

import re; import nltk;  nltk.download('stopwords');  from nltk.corpus import stopwords
from collections import Counter; from typing import List, Dict; import string; import math; import numpy as np;
# nltk.download('punkt');  nltk.download('averaged_perceptron_tagger');
import spacy

import itertools;
# len([i for i in itertools.permutations(range(0,5))])

import os;
# replace with your path..
path_to_your_files = args[1];
# path_to_your_files = "/content/drive/MyDrive/Colab Notebooks/UW/CLMS/575 - Fei/0/LING-575-project/D2/outputs" + "/training_output"
output_path = args[2];

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

"""
Extract texts data as a dictionary
"""

# Define the function to process all files in a directory
def get_structured_data_from_path(training_dir):
    # training_dir = directory+"/training_output";
    # training_dir = directory+"/devtest_output";

    data_original_backup_by_file = {}; # texts by file before cleaning

    data_original_backup_by_doc = {};  # texts aggregated by doc before cleaning

    # Walk through the directory
    for docSetA in os.listdir(training_dir):
        docSetA_dir = training_dir+"/"+docSetA;

        data_original_backup_by_file[docSetA]={};

        data_original_backup_by_doc[docSetA]=[];


        for file in os.listdir(docSetA_dir):
            file_path = os.path.join(docSetA_dir, file)

            data_original_backup_by_file[docSetA][file] = [];

            sentences_list = get_TEXT(file_path)['TEXT'];

            for sent in sentences_list:
                data_original_backup_by_file[docSetA][file].append(sent);
                data_original_backup_by_doc[docSetA].append(sent);

    return {2: data_original_backup_by_file, 3: data_original_backup_by_doc};

"""
IMPORTANT, original data and cleaned data
"""
prelim_data = get_structured_data_from_path(path_to_your_files);

"""
IMPORTANT, save spacy_nlp_results
"""

# Load the English language model
spacy_nlp = spacy.load("en_core_web_sm")

spacy_nlp_results_by_file = {};

def get_spacy_nlp_doc_results(data):
    for docSetA in data:
        spacy_nlp_results_by_file[docSetA] = {};
        for file in data[docSetA]:
            spacy_nlp_results_by_file[docSetA][file] = [];
            sentences_list = data[docSetA][file];
            for sent in sentences_list:
                spacy_nlp_results_by_file[docSetA][file].append(spacy_nlp(sent))

get_spacy_nlp_doc_results(data = prelim_data[2])



'''
helper functions
'''
def check_sentence_completeness(spacy_nlp_doc):
    has_subject = any(token.dep_.endswith("subj") for token in spacy_nlp_doc)
    # has_object  = any(token.dep_.endswith("obj") for token in spacy_nlp_doc)
    has_verb = any(token.pos_ == "VERB" for token in spacy_nlp_doc)
    contains_entity = len(spacy_nlp_doc.ents) > 0
    return has_subject and has_verb and contains_entity

# def further_filtering(spacy_nlp_doc):

#     return False;

# word processing
def process_words(sentence_as_string, spacy_nlp_doc, ngram, remove_stop_words = remove_stopwords_global):

    # must contain: subject, verb, named entity
    is_complete_sentence = check_sentence_completeness(spacy_nlp_doc);

    if not is_complete_sentence:
        return [];

    list_of_words = sentence_as_string.split();
    result  = None;

    punctuation_pattern = f'^[{re.escape(string.punctuation)}]+$'
    # punctuation_re = re.compile(r'[^\w\s]|_')
    list_of_words = [word for word in list_of_words if (not bool(re.match(punctuation_pattern, word))) and (not "'s" in word)]

    ## for removing stop words
    if remove_stop_words:
        list_of_words = [word for word in list_of_words if word.lower() not in stop_words];
    else:
        pass;

    ## for bigrams
    list_of_words.insert(0, '<s>');
    list_of_words.append('</s>');
    if ngram == 2:
        result = [(list_of_words[i], list_of_words[i + 1]) for i in range(len(list_of_words) - 1)]
    else:
        result = list_of_words[1:-1]

    if len(result) < length_cut:
        result = [];
    return result

stop_words = set([sw.lower() for sw in stopwords.words("english")])
stop_words.update(['say', 'said', 'I'])

# Define the function to process all files in a directory
def get_data_for_sumbasic(data, ngram):

    data_by_file  = {};       # texts by file after cleaning
    data_by_doc  = {};        # texts aggregated by doc after cleaning

    # Walk through the directory
    for docSetA in data:

        data_by_file[docSetA]={};
        data_by_doc[docSetA]=[];

        for file in data[docSetA]:
            data_by_file[docSetA][file] = [];
            sentences_list = data[docSetA][file];
            for i, sent in enumerate(sentences_list):
                words = process_words(sentence_as_string = sent,
                                      spacy_nlp_doc = spacy_nlp_results_by_file[docSetA][file][i],
                                      ngram = ngram);
                data_by_file[docSetA][file].append(words);
                data_by_doc[docSetA].append(words);

    prelim_data[0] = data_by_file;
    prelim_data[1] = data_by_doc;

get_data_for_sumbasic(data = prelim_data[2], ngram = ngram_global)

# Define the function to calculate word probabilities
def get_word_probabilities(data = prelim_data[0]):

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

"""
IMPORTANT, save word probability results
"""
inter_results_by_file = get_word_probabilities();
inter_results_by_doc = {k:v for k,v in inter_results_by_file.items()};
inter_results_by_doc[0] = prelim_data[1]


###########################
### get tf-idf
###########################

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

tf_idf_result_by_file = get_tfidf_by_file(prelim_data[0]);

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

tf_idf_result_by_doc = get_tfidf_by_doc(prelim_data[1]);

tf_idf_result = {'file': tf_idf_result_by_file, 'doc': tf_idf_result_by_doc}



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

'''
warning: DID use the original sentence length
'''

import numpy as np;

# Define the function to collect enough sentences to form the summary.

def get_final_sentence_scores_by_doc(data = inter_results_by_doc[0], word_probabilities = inter_results_by_doc[2],
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
        ######## current_cumulative_len = len(data[docSetA][current_top_idx[-1]]);
        current_cumulative_len = len(prelim_data[3][docSetA][current_top_idx[-1]].split()); '''use original sent length'''
        if file_length_words <= n_words:
            training_summary_n_words[docSetA] = data[docSetA];
            # training_summary_n_words[docSetA] = prelim_data[3][docSetA];
        else:
            while current_cumulative_len < n_words:
                # update the score of the top
                if word_score == 'tfidf':
                    current_doc_scores[current_top_idx[-1]] = calculate_sentence_score_tfidf_by_doc(data[docSetA][current_top_idx[-1]], tfidf['doc'][docSetA], power = 2);
                else:
                    current_doc_scores[current_top_idx[-1]] = calculate_sentence_score_prob(data[docSetA][current_top_idx[-1]], word_probabilities, power = 2);
                #################################
                # update the cumulative length
                if data[docSetA][np.argmax(current_doc_scores)] == data[docSetA][current_top_idx[-1]]:
                    current_doc_scores[np.argmax(current_doc_scores)] = calculate_sentence_score_prob(data[docSetA][np.argmax(current_doc_scores)], word_probabilities, power = 2);
                    # if not the above line, dead loop;
                    continue;
                # if np.argmax(current_doc_scores) == current_top_idx[-1]:
                #     # does not work! because these two sentences appear in different files!
                #     continue;
                ### current_cumulative_len += len(data[docSetA][np.argmax(current_doc_scores)])
                current_cumulative_len += len(prelim_data[3][docSetA][np.argmax(current_doc_scores)].split()); '''use original sent length'''
                #####################################
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

        summary_n_words_cleaned[docSetA] = [prelim_data[1][docSetA][idx] for idx in arg['n_words'][docSetA]];
        summary_n_words[docSetA]         = [prelim_data[3][docSetA][idx] for idx in arg['n_words'][docSetA]]

    return {"n_words": {0: summary_n_words_cleaned,1:summary_n_words}}

# preliminary sentence scores
sumBasic_by_doc = get_sumBasic_by_doc()

def write_to_file():
    for docSetA in sumBasic_by_doc['n_words'][1]:
        part1 = docSetA[:-3];
        part2 = docSetA[-3];
        name_file = part1 + "-A.M.100." + part2 + ".2";
        # with open(f'{output_path}/D4/sumbasic_summaries_no_stopwords/{name_file}', 'w') as file:
        if remove_stopwords_global == True:
            dir = "sumbasic_summaries_no_stopwords";
        else:
            dir = "sumbasic_summaries_stopwords";
        with open(f'{output_path}/D4/{dir}/{name_file}', 'w') as file:
            for line in sumBasic_by_doc['n_words'][1][docSetA]:
                file.write(f"{line}\n")
write_to_file()
# Based of https://github.com/crabcamp/lexrank.git
from functools import partial
import math
from collections import Counter, defaultdict
import os
import re
import sys
import traceback
import numpy as np
import nltk
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from scipy.sparse.csgraph import connected_components
from LLR import get_embedding, good_sentence

nltk.download('stopwords')

def clean(file_path, add_sentence_marker=False):
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
            #import pdb; pdb.set_trace()

            # Identify the section based on flags and content
            if stripped_line.startswith('HEADLINE:'):
                is_headline = True
                cleaned_info["HEADLINE"] += stripped_line[len('HEADLINE:'):].strip() + " "
            elif stripped_line.startswith('DATELINE:') or stripped_line.startswith('DATE_TIME:'):
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
                if punctuation_re.search(stripped_line[-1:]):  # Check if the line ends with a specified punctuation
                    cleaned_sentence = re.sub(r'[^\w\s-]|_', '', stripped_line).strip()
                    if cleaned_sentence:
                        if add_sentence_marker:
                            cleaned_sentence = '<s> ' + cleaned_sentence + ' </s>'
                        cleaned_info["TEXT"].append(cleaned_sentence)
                elif stripped_line == "":
                  continue
                else:
                    # If the line doesn't end with punctuation, it's a part of the next sentence
                    if cleaned_info["TEXT"]:
                        cleaned_info["TEXT"][-1] += ' ' + re.sub(r'[^\w\s-]|_', '', stripped_line)
                    else:
                        # If TEXT is empty, start the first sentence
                        cleaned_info["TEXT"].append('<s> ' if add_sentence_marker else '' + re.sub(r'[^\w\s-]|_', '', stripped_line))

    # Remove any trailing whitespace from HEADLINE and DATELINE
    cleaned_info["HEADLINE"] = cleaned_info["HEADLINE"].strip()
    cleaned_info["DATELINE"] = cleaned_info["DATELINE"].strip()

    # Add the end sentence marker to the last sentence in TEXT
    if cleaned_info["TEXT"] and add_sentence_marker:
      if '</s>' not in cleaned_info["TEXT"][-1]:
        cleaned_info["TEXT"][-1] += ' </s>'

    return cleaned_info

def create_markov_matrix_discrete(weights_matrix, threshold):
    if threshold:
        discrete_weights_matrix = np.zeros(weights_matrix.shape)
        ixs = np.where(weights_matrix >= threshold)
        discrete_weights_matrix[ixs] = 1
    else:
        discrete_weights_matrix = weights_matrix
    
    n_1, n_2 = discrete_weights_matrix.shape
    if n_1 != n_2:
        raise ValueError('\'weights_matrix\' should be square')

    row_sum = discrete_weights_matrix.sum(axis=1, keepdims=True)

    return discrete_weights_matrix / row_sum

def connected_nodes(matrix):
    _, labels = connected_components(matrix)

    groups = []

    for tag in np.unique(labels):
        group = np.where(labels == tag)[0]
        groups.append(group)

    return groups

def _power_method(transition_matrix, increase_power=True):
    eigenvector = np.ones(len(transition_matrix))

    if len(eigenvector) == 1:
        return eigenvector

    transition = transition_matrix.transpose()

    while True:
        eigenvector_next = np.dot(transition, eigenvector)

        if np.allclose(eigenvector_next, eigenvector):
            return eigenvector_next

        eigenvector = eigenvector_next

        if increase_power:
            transition = np.dot(transition, transition)

def stationary_distribution(
    transition_matrix,
    increase_power=True,
    normalized=True,
):
    n_1, n_2 = transition_matrix.shape
    if n_1 != n_2:
        raise ValueError('\'transition_matrix\' should be square')

    distribution = np.zeros(n_1)

    grouped_indices = connected_nodes(transition_matrix)

    for group in grouped_indices:
        t_matrix = transition_matrix[np.ix_(group, group)]
        eigenvector = _power_method(t_matrix, increase_power=increase_power)
        distribution[group] = eigenvector

    if normalized:
        distribution /= n_1

    return distribution

def degree_centrality_scores(
    similarity_matrix,
    threshold,
    increase_power=True,
):
    if not (
        threshold is None
        or isinstance(threshold, float)
        and 0 <= threshold < 1
    ):
        raise ValueError(
            '\'threshold\' should be a floating-point number '
            'from the interval [0, 1) or None',
        )

    markov_matrix = create_markov_matrix_discrete(
        similarity_matrix,
        threshold,
    )

    scores = stationary_distribution(
        markov_matrix,
        increase_power=increase_power,
        normalized=False,
    )

    return scores


class LexRank:
    def __init__(
        self,
        documents,
        stopwords=None,
        keep_numbers=False,
        keep_emails=False,
        keep_urls=False,
        include_new_words=True,
    ):
        if stopwords is None:
            self.stopwords = set()
        else:
            self.stopwords = stopwords

        self.keep_numbers = keep_numbers
        self.keep_emails = keep_emails
        self.keep_urls = keep_urls
        self.include_new_words = include_new_words

        self.idf_score = self._calculate_idf(documents)

    def get_summary(
        self,
        sentences,
        summary_size=1,
        threshold=.03,
        fast_power_method=True,
    ):
        if not isinstance(summary_size, int) or summary_size < 1:
            raise ValueError('\'summary_size\' should be a positive integer')

        lex_scores = self.rank_sentences(
            sentences,
            threshold=threshold,
            fast_power_method=fast_power_method,
        )

        sorted_ix = np.argsort(lex_scores)[::-1]
        summary = [sentences[i] for i in sorted_ix[:summary_size]]

        return summary

    def rank_sentences(
        self,
        sentences,
        threshold=.03,
        fast_power_method=True,
    ):

        similarity_matrix = self._calculate_similarity_matrix(sentences)
        

        scores = degree_centrality_scores(
            similarity_matrix,
            threshold=threshold,
            increase_power=fast_power_method,
        )

        return scores

    def sentences_similarity(self, sentence_1, sentence_2):
        tf_1 = Counter(self.tokenize_sentence(sentence_1))
        tf_2 = Counter(self.tokenize_sentence(sentence_2))

        similarity = self._idf_modified_cosine([tf_1, tf_2], 0, 1)

        return similarity

    def tokenize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        cleanup = list(filter(lambda word: word not in self.stopwords, tokens))
        return cleanup

    def _calculate_idf(self, documents):
        bags_of_words = []

        for doc in documents:
            doc_words = set()

            for sentence in doc:
                words = self.tokenize_sentence(sentence)
                doc_words.update(words)

            if doc_words:
                bags_of_words.append(doc_words)

        if not bags_of_words:
            raise ValueError('documents are not informative')

        doc_number_total = len(bags_of_words)

        if self.include_new_words:
            default_value = 1

        else:
            default_value = 0

        idf_score = defaultdict(lambda: default_value)

        for word in set.union(*bags_of_words):
            doc_number_word = sum(1 for bag in bags_of_words if word in bag)
            idf_score[word] = math.log(doc_number_total / doc_number_word)

        return idf_score

    def _calculate_similarity_matrix(self, sentences):
        embeddings = [get_embedding(sentence) for sentence in sentences]
        length = len(sentences)

        similarity_matrix = np.zeros([length] * 2)

        for i in range(length):
            for j in range(i, length):
                similarity = self._embedding_cosine(embeddings, i, j)

                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix
    
    def _embedding_cosine(self, embeddings, i, j):
        if i == j:
            return 1
        
        embedding_i, embedding_j = embeddings[i], embeddings[j]
        return 1 - cosine(embedding_i, embedding_j)
        
        

    def _idf_modified_cosine(self, tf_scores, i, j):
        if i == j:
            return 1

        tf_i, tf_j = tf_scores[i], tf_scores[j]
        words_i, words_j = set(tf_i.keys()), set(tf_j.keys())

        nominator = 0

        for word in words_i & words_j:
            idf = self.idf_score[word]
            nominator += tf_i[word] * tf_j[word] * idf ** 2

        if math.isclose(nominator, 0):
            return 0

        denominator_i, denominator_j = 0, 0

        for word in words_i:
            tfidf = tf_i[word] * self.idf_score[word]
            denominator_i += tfidf ** 2

        for word in words_j:
            tfidf = tf_j[word] * self.idf_score[word]
            denominator_j += tfidf ** 2

        similarity = nominator / math.sqrt(denominator_i * denominator_j)

        return similarity

#path_to_your_files = "/home2/jiayiy9/ling575/LING-575-project/D2/outputs"
#test_path = path_to_your_files + "/training_output/D0901A-A/AFP_ENG_20050119.0019"

def sentence_filter(sentence, sentence_length_threshold):
    words = [word.lower() for word in sentence.split()]
    return len(words) > sentence_length_threshold and good_sentence(" ".join(words))

if __name__ == "__main__":
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    sentence_length_threshold = int(sys.argv[3])
    similarity_threshold = float(sys.argv[4])
    summary_length = int(sys.argv[5])
    en_stop_words = set(stopwords.words("english"))
    
    os.makedirs(output_directory, exist_ok=True)
    
    for subdir in os.listdir(input_directory):
        files_path = os.path.join(input_directory, subdir)
        sentences = []
        for file_name in os.listdir(files_path):
            file_path = os.path.join(files_path, file_name)
            cleaned_text_dict = clean(file_path)
            sentences.extend(cleaned_text_dict['TEXT'])
        
        sentences = list(filter(partial(sentence_filter, sentence_length_threshold=sentence_length_threshold), sentences))
            
        try:
            lxr = LexRank(sentences, stopwords=en_stop_words)
            summary = lxr.get_summary(sentences, summary_size=summary_length, threshold=similarity_threshold)
            filename = f"{subdir[:-3]}-A.M.100.{subdir[-3]}.3"
            with open(os.path.join(output_directory, filename), "w") as output:
                output.write("\n".join(summary))
                
        except Exception:
            print(traceback.format_exc())
            print(cleaned_text_dict)
            import pdb; pdb.set_trace()
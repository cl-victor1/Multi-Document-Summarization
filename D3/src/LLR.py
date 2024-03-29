import math
import sys
from collections import defaultdict
import os
from decimal import Decimal
from nltk.corpus import stopwords
import string

def combinations(n, k):
    # Calculate n! / (k! * (n - k)!)
    # This approach avoids computing large factorials and mitigates the risk of overflow errors.
    result = 1
    for i in range(1, k + 1):
        result *= (n - i + 1) / i
    return result

def background_count(back_corpus_file, punctuation, stopwords):
    word_count = defaultdict(int)
    total_words = 0
    with open(back_corpus_file, 'r') as back:
        for line in back:
            words = [word.lower() for word in line.strip().split()] # ignore the case
            for word in words:
                if word not in punctuation: # exclude punctuations
                    if stopwords:
                        if word not in stopwords:
                            total_words += 1
                            word_count[word] += 1 
                    else:
                        total_words += 1
                        word_count[word] += 1 
    return total_words, word_count
         
def input_count(files_path, punctuation, stopwords):         
    word_count = defaultdict(int)
    total_words = 0
    files = os.listdir(files_path)
    all_sentences = []
    # Iterate through each file
    for file_name in files:
        # Create the full path to the file
        file_path = os.path.join(files_path, file_name)
        with open(file_path, 'r') as file:
            for line in file.readlines():
                # exclude meta information
                if not line.startswith("HEADLINE") and not line.startswith("DATE_TIME") \
                    and not line.startswith("DATETIME") and not line.startswith("DATELINE"):
                    words = [word.lower() for word in line.strip().split()] # ignore the case
                    for word in words:
                        if word not in punctuation:
                            if stopwords:
                                if word not in stopwords:
                                    total_words += 1
                                    word_count[word.lower()] += 1 # ignore the case
                            else:
                                total_words += 1
                                word_count[word.lower()] += 1 # ignore the case
                    if len(words) > 5: # exclude sentences whose length is <= 5
                        all_sentences.append(words)                        
    return total_words, word_count, all_sentences      
            
def LLR(n2, back_word_count, n1, input_word_count, confidence_level):
    important_words = set()
    for word in input_word_count.keys():
        k1 = input_word_count[word]
        k2 = max(back_word_count[word], 1) # to avoid log(0)
        p1 = k1 / n1
        p2 = k2 / n2
        p = (k1 + k2) / (n1 + n2)
        # change multiplications into additions of log
        ratio = 2 * (math.log(Decimal(combinations(n1, k1))) + k1 * math.log(p1) + (n1 - k1) * math.log(1 - p1)\
            + math.log(Decimal(combinations(n2, k2))) + k2 * math.log(p2) + (n2 - k2) * math.log(1 - p2)\
            - math.log(Decimal(combinations(n1, k1))) - k1 * math.log(p) - (n1 - k1) * math.log(1 - p)\
                - math.log(Decimal(combinations(n2, k2))) - k2 * math.log(p) - (n2 - k2) * math.log(1 - p))
        if float(ratio) > confidence_level:
            important_words.add(word)
    return important_words
        
def calculate_weight(sentence, important_words): # every sentence is a list of tokenized words
    weight = 0 # weight is the number of important words that each sentence has
    sentence_length = 0
    for word in sentence:
        sentence_length += 1
        if word in important_words:
            weight += 1
    return weight / sentence_length # weight of the sentence
    
        
if __name__ == "__main__":   
    back_corpus_file = sys.argv[1]
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]
    confidence_level = float(sys.argv[4])
    summary_length = int(sys.argv[5])
    if_stopwords = int(sys.argv[6]) # 0 means not using stopwords, 1 means using stopwords
    
    english_punctuation = set(string.punctuation)    
    english_stopwords = None
    if if_stopwords == 1:
        english_stopwords = set(stopwords.words('english'))
        
    back_total_words, back_word_count = background_count(back_corpus_file, english_punctuation, english_stopwords)        
    for subdir in os.listdir(input_directory):
        files_path = os.path.join(input_directory, subdir)
        input_total_words, input_word_count, all_sentences = input_count(files_path, english_punctuation, english_stopwords)
        important_words = LLR(back_total_words, back_word_count, input_total_words, input_word_count, confidence_level)
        sentence_weights = {} # keep track of weight of all sentences
        for i in range(len(all_sentences)): 
            # keep track of length and weight of each sentence
            sentence_weights[i] = (len(all_sentences[i]), calculate_weight(all_sentences[i], important_words))
        ordered_sentences = sorted(sentence_weights.items(), key=lambda x: x[1][1])
        selected_sentences_indices = []
        curr_length = 0
        while True:
            if ordered_sentences == []:
                break
            chosen = ordered_sentences.pop() # chose the most weighted sentence
            if curr_length + chosen[1][0] <= summary_length:
                selected_sentences_indices.append(chosen[0]) # append index
                curr_length += chosen[1][0]   # length
            else:
                break
        
        # Create the output_directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        filename = f"{subdir[:-3]}-A.M.100.{subdir[-3]}.1"
        with open(os.path.join(output_directory, filename), "w") as output:
            for i in selected_sentences_indices:
                output.write(" ".join(all_sentences[i]) + "\n")
        
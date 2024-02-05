"""
This is the sample code for implementing ROUGE using the package rouge-score (https://pypi.org/project/rouge-score/).
"""
import os
import sys

from rouge_score import rouge_scorer

def rouge(target_file_dir, prediction_file_dir, output_filename):
    target_filenames = os.listdir(target_file_dir)
    prediction_filenames = os.listdir(prediction_file_dir)


    for filename in target_filenames:
        target_filename = str(target_file_dir + filename)
        if target_file_dir.startswith("LLR"):
            prediction_filename = str(prediction_file_dir + "/" + filename)
        else:
            prediction_filename = str(prediction_file_dir + filename)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        target_file = open(target_filename, "r")
        prediction_file = open(prediction_filename, "r")

        target_sentences_list = target_file.readlines()
        target_summary = "".join(target_sentences_list)
        prediction_sentences_list = prediction_file.readlines()
        prediction_summary = " ".join(prediction_sentences_list)

        output_file = open(output_filename, "a+")
        score = scorer.score(target_summary, prediction_summary)
        output_file.writelines("ROUGE for " + str(target_filename)[-17:] + " is: " + str(score) + "\n")


if __name__ == "__main__":
    target_file_directory = sys.argv[1]
    prediction_file_directory = sys.argv[2]
    output_filename = sys.argv[3]

    rouge(target_file_directory, prediction_file_directory, output_filename)

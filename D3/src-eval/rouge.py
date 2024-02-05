"""
This is the sample code for implementing ROUGE using the package rouge-score (https://pypi.org/project/rouge-score/).
"""
import os
import sys

from rouge_score import rouge_scorer

def rouge(target_file_dir, prediction_file_dir, output_filename):
    target_filenames = os.listdir(target_file_dir)
    prediction_filenames = os.listdir(prediction_file_dir)
    output_file = open(output_filename, "a+")

    rouge1_precisions = []
    rouge1_recalls = []
    rouge1_fmeasures = []

    rouge2_precisions = []
    rouge2_recalls = []
    rouge2_fmeasures = []

    rouge3_precisions = []
    rouge3_recalls = []
    rouge3_fmeasures = []

    rouge4_precisions = []
    rouge4_recalls = []
    rouge4_fmeasures = []

    rougeL_precisions = []
    rougeL_recalls = []
    rougeL_fmeasures = []

    for filename in target_filenames:
        target_filename = str(target_file_dir + filename)
        # if target_file_dir.startswith("../outputs/LLR"):
        prediction_filename = str(prediction_file_dir + filename)
        #prediction_filename = str(prediction_file_dir + filename)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL'], use_stemmer=True)

        target_file = open(target_filename, "r")
        prediction_file = open(prediction_filename, "r")

        target_sentences_list = target_file.readlines()
        target_summary = "".join(target_sentences_list)
        prediction_sentences_list = prediction_file.readlines()
        prediction_summary = " ".join(prediction_sentences_list)


        score = scorer.score(target_summary, prediction_summary)

        rouge1_precisions.append(score["rouge1"].precision)
        rouge1_recalls.append(score["rouge1"].recall)
        rouge1_fmeasures.append(score["rouge1"].fmeasure)

        rouge2_precisions.append(score["rouge2"].precision)
        rouge2_recalls.append(score["rouge2"].recall)
        rouge2_fmeasures.append(score["rouge2"].fmeasure)

        rouge3_precisions.append(score["rouge3"].precision)
        rouge3_recalls.append(score["rouge3"].recall)
        rouge3_fmeasures.append(score["rouge3"].fmeasure)

        rouge4_precisions.append(score["rouge4"].precision)
        rouge4_recalls.append(score["rouge4"].recall)
        rouge4_fmeasures.append(score["rouge4"].fmeasure)

        rougeL_precisions.append(score["rougeL"].precision)
        rougeL_recalls.append(score["rougeL"].recall)
        rougeL_fmeasures.append(score["rougeL"].fmeasure)


        output_file.writelines("ROUGE for " + str(target_filename)[-17:] + " is: " + str(score) + "\n")
    output_file.writelines("------------\n")
    output_file.writelines("ROUGE-1 Average R: " + str(sum(rouge1_recalls)/len(rouge1_recalls)) + "\n")
    output_file.writelines("ROUGE-1 Average P: " + str(sum(rouge1_precisions)/len(rouge1_precisions)) + "\n")
    output_file.writelines("ROUGE-1 Average F: " + str(sum(rouge1_fmeasures)/len(rouge1_fmeasures)) + "\n")
    output_file.writelines("------------\n")
    output_file.writelines("ROUGE-2 Average R: " + str(sum(rouge2_recalls) / len(rouge2_recalls)) + "\n")
    output_file.writelines("ROUGE-2 Average P: " + str(sum(rouge2_precisions) / len(rouge2_precisions)) + "\n")
    output_file.writelines("ROUGE-2 Average F: " + str(sum(rouge2_fmeasures) / len(rouge2_fmeasures)) + "\n")
    output_file.writelines("------------\n")
    output_file.writelines("ROUGE-3 Average R: " + str(sum(rouge3_recalls) / len(rouge3_recalls)) + "\n")
    output_file.writelines("ROUGE-3 Average P: " + str(sum(rouge3_precisions) / len(rouge3_precisions)) + "\n")
    output_file.writelines("ROUGE-3 Average F: " + str(sum(rouge3_fmeasures) / len(rouge3_fmeasures)) + "\n")
    output_file.writelines("------------\n")
    output_file.writelines("ROUGE-4 Average R: " + str(sum(rouge4_recalls) / len(rouge4_recalls)) + "\n")
    output_file.writelines("ROUGE-4 Average P: " + str(sum(rouge4_precisions) / len(rouge4_precisions)) + "\n")
    output_file.writelines("ROUGE-4 Average F: " + str(sum(rouge4_fmeasures) / len(rouge4_fmeasures)) + "\n")
    output_file.writelines("------------\n")
    output_file.writelines("ROUGE-L Average R: " + str(sum(rougeL_recalls) / len(rougeL_recalls)) + "\n")
    output_file.writelines("ROUGE-L Average P: " + str(sum(rougeL_precisions) / len(rougeL_precisions)) + "\n")
    output_file.writelines("ROUGE-L Average F: " + str(sum(rougeL_fmeasures) / len(rougeL_fmeasures)) + "\n")

if __name__ == "__main__":
    target_file_directory = sys.argv[1]
    prediction_file_directory = sys.argv[2]
    output_filename = sys.argv[3]

    rouge(target_file_directory, prediction_file_directory, output_filename)

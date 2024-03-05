import os
import sys
import pandas as pd
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer


def calc_avg_document_scores(target_file_dir, target_filename_unspecified, prediction_filename):
    target_filenames = os.listdir(target_file_dir)
    prediction_file = open(prediction_filename, "r")
    target_file_vers = ["A", "B", "C", "D", "E", "F", "G", "H"]
    pearson_scores = []
    model = SentenceTransformer("../data/all-MiniLM-L6-v2")
    # calculate pearson correlation score between a prediction summary
    # with respect to EACH version of human summary
    for ver in target_file_vers:
        filename_check = str(target_filename_unspecified + ver)
        if filename_check in target_filenames:
            print("current target file found in target dir: " + filename_check)

            # import target summaries and prediction summaries
            with open(target_file_dir + filename_check, encoding="ISO-8859-1") as t:
                target_sentences_list = t.readlines()
            target_summary = "".join(target_sentences_list)
            prediction_sentences_list = prediction_file.readlines()
            prediction_summary = " ".join(prediction_sentences_list)

            # preprocess text data into numerical values / embeddings
            target_summary_numerical = model.encode(target_summary)
            prediction_summary_numerical = model.encode(prediction_summary)

            # append each pearson score to the same list
            corr, _ = pearsonr(target_summary_numerical, prediction_summary_numerical)
            pearson_scores.append(corr)
        else:
            print("current target file NOT found in target dir: " + filename_check)
            continue

    return sum(pearson_scores)/len(pearson_scores)


def pearson(target_file_dir, prediction_file_dir, method_id, output_csv_filename):
    prediction_filenames = os.listdir(prediction_file_dir)

    avg_pearson_scores = []

    for filename in prediction_filenames:
        filename = filename[:-1]
        prediction_filename = str(prediction_file_dir + filename + method_id)
        target_filename_unspecified = str(filename)
        avg_pearson_scores.append(calc_avg_document_scores(
            target_file_dir, target_filename_unspecified, prediction_filename))

    result_dict = {"pearson_scores": avg_pearson_scores}
    df = pd.DataFrame(result_dict)
    df.to_csv(str(output_csv_filename))

if __name__ == "__main__":
    target_file_directory = sys.argv[1]
    prediction_file_directory = sys.argv[2]
    method_id = sys.argv[3]
    output_csv_filename = sys.argv[4]

    pearson(target_file_directory, prediction_file_directory, method_id, output_csv_filename)
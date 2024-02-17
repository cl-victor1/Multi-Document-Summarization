import os
import sys
import pandas as pd

from torchmetrics.functional.text.infolm import infolm

def info_lm(target_file_dir, prediction_file_dir, output_csv_filename):
    target_filenames = os.listdir(target_file_dir)

    kl_list, l1_list, l2_list, l_inf_list = [], [], [], []

    for filename in target_filenames:
        target_filename = str(target_file_dir + filename)
        prediction_filename = str(prediction_file_dir + filename)

        target_file = open(target_filename, "r")
        prediction_file = open(prediction_filename, "r")

        target_sentences_list = target_file.readlines()
        target_summary = "".join(target_sentences_list)
        prediction_sentences_list = prediction_file.readlines()
        prediction_summary = " ".join(prediction_sentences_list)

        kl_divergence_score = infolm(prediction_summary, target_summary, model_name_or_path='../data/bert_uncased_L-12_H-768_A-12',
                                             idf=False, information_measure="kl_divergence")
        l1_distance_score = infolm(prediction_summary, target_summary,
                                            model_name_or_path='../data/bert_uncased_L-12_H-768_A-12',
                                            idf=False, information_measure="l1_distance")
        l2_distance_score = infolm(prediction_summary, target_summary,
                                   model_name_or_path='../data/bert_uncased_L-12_H-768_A-12',
                                   idf=False, information_measure="l2_distance")
        l_inf_distance_score = infolm(prediction_summary, target_summary,
                                   model_name_or_path='../data/bert_uncased_L-12_H-768_A-12',
                                   idf=False, information_measure="l_infinity_distance")

        print("Processing file" + str(filename))
        print(str(kl_divergence_score))
        print(str(l1_distance_score))
        print(str(l2_distance_score))
        print(str(l_inf_distance_score))

        kl_list.append(str(kl_divergence_score))
        l1_list.append(str(l1_distance_score))
        l2_list.append(str(l2_distance_score))
        l_inf_list.append(str(l_inf_distance_score))

    result_dict = {"KL_divergence_scores": kl_list, "L1": l1_list, "L2": l2_list, "L_inf":l_inf_list}
    df = pd.DataFrame(result_dict)
    df.to_csv(str(output_csv_filename))

if __name__ == "__main__":
    target_file_directory = sys.argv[1]
    prediction_file_directory = sys.argv[2]
    output_csv_filename = sys.argv[3]

    info_lm(target_file_directory, prediction_file_directory, output_csv_filename)
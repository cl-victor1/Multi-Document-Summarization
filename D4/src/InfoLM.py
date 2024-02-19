import os
import sys
import pandas as pd

from torchmetrics.functional.text.infolm import infolm

<<<<<<< Updated upstream
def calc_avg_document_scores(target_file_dir, target_filename_unspecified, prediction_filename):
    target_filenames = os.listdir(target_file_dir)
    prediction_file = open(prediction_filename, "r")

    plm_path = "../data/bert_uncased_L-12_H-768_A-12" # path to the pretrained lang model
    target_file_vers = ["A", "B", "C", "D", "E", "F", "G", "H"]

    kl_scores, l1_scores, l2_scores, l_inf_scores = [], [], [], []

    for ver in target_file_vers:
        filename_check = str(target_filename_unspecified + ver)
        if filename_check in target_filenames:
            print("current target file found in target dir: " + filename_check)
            with open(target_file_dir + filename_check, encoding="ISO-8859-1") as t:
                target_sentences_list = t.readlines()
            target_summary = "".join(target_sentences_list)
            prediction_sentences_list = prediction_file.readlines()
            prediction_summary = " ".join(prediction_sentences_list)

            kl_score = infolm(prediction_summary, target_summary, model_name_or_path=plm_path, idf=False,
                              information_measure="kl_divergence")
            l1_score = infolm(prediction_summary, target_summary, model_name_or_path=plm_path, idf=False,
                              information_measure="l1_distance")
            l2_score = infolm(prediction_summary, target_summary, model_name_or_path=plm_path, idf=False,
                              information_measure="l2_distance")
            l_inf_score = infolm(prediction_summary, target_summary, model_name_or_path=plm_path, idf=False,
                                 information_measure="l_infinity_distance")
            kl_scores.append(kl_score)
            l1_scores.append(l1_score)
            l2_scores.append(l2_score)
            l_inf_scores.append(l_inf_score)
        else:
            print("current target file NOT found in target dir: " + filename_check)
            continue

    return (sum(kl_scores)/len(kl_scores), sum(l1_scores)/len(l1_scores), sum(l2_scores)/len(l2_scores), \
            sum(l_inf_scores)/len(l_inf_scores))
    
def info_lm(target_file_dir, prediction_file_dir, method_id, output_csv_filename):
    prediction_filenames = os.listdir(prediction_file_dir)

    kl_list, l1_list, l2_list, l_inf_list = [], [], [], [] # store scores of each document

    for filename in prediction_filenames:
        filename = filename[:-1]
        prediction_filename = str(prediction_file_dir + filename + method_id)
        target_filename_unspecified = str(filename)
        avg_kl_score, avg_l1_score, avg_l2_score, avg_l_inf_score = calc_avg_document_scores(
            target_file_dir, target_filename_unspecified, prediction_filename)

        print("Processing prediction file: " + str(filename))
        print(str(avg_kl_score))
        print(str(avg_l1_score))
        print(str(avg_l2_score))
        print(str(avg_l_inf_score))

        kl_list.append(str(avg_kl_score))
        l1_list.append(str(avg_l1_score))
        l2_list.append(str(avg_l2_score))
        l_inf_list.append(str(avg_l_inf_score))

    result_dict = {"KL_divergence_scores": kl_list, "L1": l1_list, "L2": l2_list, "L_inf": l_inf_list}
=======
def info_lm(target_file_dir, prediction_file_dir, output_csv_filename):
    prediction_filenames = os.listdir(prediction_file_dir)

    kl_list, l1_list, l2_list, l_inf_list = [], [], [], []

    for filename in prediction_filenames:
        target_filename = str(target_file_dir + filename[:-1] + "A")
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

        print("Processing target file " + str(target_filename) + " and prediction file " + str(prediction_filename))
        print(str(kl_divergence_score))
        print(str(l1_distance_score))
        print(str(l2_distance_score))
        print(str(l_inf_distance_score))

        kl_list.append(str(kl_divergence_score))
        l1_list.append(str(l1_distance_score))
        l2_list.append(str(l2_distance_score))
        l_inf_list.append(str(l_inf_distance_score))

    result_dict = {"KL_divergence_scores": kl_list, "L1": l1_list, "L2": l2_list, "L_inf":l_inf_list}
>>>>>>> Stashed changes
    df = pd.DataFrame(result_dict)
    df.to_csv(str(output_csv_filename))


if __name__ == "__main__":
    target_file_directory = sys.argv[1]
    prediction_file_directory = sys.argv[2]
    method_id = sys.argv[3]
    output_csv_filename = sys.argv[4]

    info_lm(target_file_directory, prediction_file_directory, method_id, output_csv_filename)
#!/bin/sh

# sumBasic
python ./sumbasic.py /D2/outputs/devtest_output /D4/outputs/sumbasic_summaries_no_stopwords -ry -1 prob 5
python ./sumbasic.py /D2/outputs/devtest_output /D4/outputs/sumbasic_summaries_stopwords -rn -1 prob 5
python ./info_ordering_entity_grid.py /D2/outputs/devtest_output /D4/outputs/sumbasic_summaries_no_stopwords /D4/outputs/sumbasic_summaries_final

# LLR
python ./make_LLR_background.py /D4/data/back_corpus_file /D4/data/back_corpus
python ./LLR.py /D4/data/back_corpus_file /D2/outputs/devtest_output /D4/outputs/LLR_summaries_no_stopwords_0.05 0.05 100 0 5 0.8
python ./LLR.py /D4/data/back_corpus_file /D2/outputs/devtest_output /D4/outputs/LLR_summaries_stopwords_0.05 0.05 100 1 5 0.8

# lexrank
python ./lexrank.py /D2/outputs/devtest_output /D4/outputs/LexRank_sentence_length_10_threshold_0.3_summary_length_5_chronological_ordering 10 0.3 5 chronological

# evaluation
python3 ./rouge_eval.py /D4/data/model_devtest /D4/outputs/LexRank_sentence_length_10_threshold_0.3_summary_length_5 > rouge_scores_LexRank.out
python3 ./rouge_eval.py /D4/data/model_devtest /D4/outputs/sumbasic_summaries_final > rouge_scores_sumBasic.out
python3 ./rouge_eval.py /D4/data/model_devtest /D4/outputs/LLR_summaries_no_stopwords_0.05 > rouge_scores_LLR.out

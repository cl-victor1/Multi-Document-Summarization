#!/bin/sh
pip install rouge-score

# LLR
python ./make_LLR_background.py /D3/data/back_corpus_file /D3/data/back_corpus
python ./LLR.py /D3/data/back_corpus_file /D2/outputs/devtest_output /D3/outputs/LLR_summaries_no_stopwords_0.05 0.05 100 0
python ./LLR.py /D3/data/back_corpus_file /D2/outputs/devtest_output /D3/outputs/LLR_summaries_stopwords_0.05 0.05 100 1
# sumBasic
python ./sumBasic.py /D2/outputs -ry -1
python ./sumBasic.py /D2/outputs -rn -1
python ./sumBasic.py /D2/outputs -rn -2
# lexrank
python ./lexrank.py /D2/outputs/devtest_output /D3/outputs/LexRank_summary_length_5 5
# evaluation
python ./evaluation.py /D3/data/model_devtest /D3/outputs/LLR_summaries_no_stopwords_0.05 > /D3/results/rouge_scores_LLR_noStop.out
python ./evaluation.py /D3/data/model_devtest /D3/outputs/LLR_summaries_stopwords_0.05 > /D3/results/rouge_scores_LLR_withStop.out
python ./evaluation.py /D3/data/model_devtest /D3/outputs/sumbasic_unigram_withStop_probability > /D3/results/rouge_scores_sumBasic_unigram_withStop.out
python ./evaluation.py /D3/data/model_devtest /D3/outputs/sumbasic_unigram_noStop_probability > /D3/results/rouge_scores_sumBasic_unigram_noStop.out
python ./evaluation.py /D3/data/model_devtest /D3/outputs/sumbasic_bigram_withStop_probability > /D3/results/rouge_scores_sumBasic_bigram_withStop.out
python ./evaluation.py /D3/data/model_devtest /D3/outputs/LexRank_summary_length_5 > /D3/results/rouge_scores_LexRank.out
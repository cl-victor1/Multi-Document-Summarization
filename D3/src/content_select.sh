#!/bin/sh
pip install rouge-score

# make background corpus for LLR
python ./src/make_LLR_background.py D3/data/back_corpus_file D3/data/back_corpus
# with and without stopwords
python ./src/LLR.py D3/data/back_corpus_file D2/outputs/devtest_output D3/outputs/LLR_summaries_no_stopwords_0.05 0.05 100 0
python ./src/LLR.py D3/data/back_corpus_file D2/outputs/devtest_output D3/outputs/LLR_summaries_stopwords_0.05 0.05 100 1

python ./src/sumBasic.py ？
python ./src/sumBasic.py ？
python ./src/sumBasic.py ？

python ./src/lexrank.py D2/outputs/devtest_output D3/outputs/lexrank_summaries 100

python ./src/evaluation.py D3/data/model_devtest D3/outputs/LLR_summaries_no_stopwords_0.05 > D3/results/rouge_scores_LLR.out
python ./src/evaluate.py D3/data/model_devtest D3/outputs/LLR_summaries_stopwords_0.05 ???
python ./src/evaluate.py D3/data/model_devtest D3/outputs/sumbasic_unigram_withStop_probability > D3/results/rouge_scores_sumBasic_unigram_withStop.out
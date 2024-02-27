#!/bin/sh

# LLR
python ./make_LLR_background.py /D5/data/back_corpus_file /D5/data/back_corpus
python ./LLR.py /D5/data/back_corpus_file /D2/outputs/devtest_output /D5/outputs/LLR_summaries_no_stopwords_0.05 0.05 100 0 5 0.8
python ./LLR.py /D5/data/back_corpus_file /D2/outputs/devtest_output /D5/outputs/LLR_summaries_stopwords_0.05 0.05 100 1 5 0.8
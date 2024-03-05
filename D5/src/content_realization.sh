#!/bin/sh

# baseline
python ./baseline.py /D2/outputs/evaltest_output /D5/outputs/baseline_evaltest 100

# LLR
python ./make_LLR_background.py /D5/data/back_corpus_file /D5/data/back_corpus
# outputs before content realization and removed from /D5/outputs
python ./LLR.py /D5/data/back_corpus_file /D2/outputs/devtest_output /D5/outputs/LLR_devtest_0.05 0.05 100 0 5 0.8
python ./LLR.py /D5/data/back_corpus_file /D2/outputs/evaltest_output /D5/outputs/LLR_evaltest_0.05 0.05 100 0 5 0.8
# outputs after content realization
python ./content_realization.py /D5/outputs/LLR_devtest_0.05 /D5/outputs/LLR_devtest
python ./content_realization.py /D5/outputs/LLR_evaltest_0.05 /D5/outputs/LLR_evaltest

# sumbasic
python ./sumbasic.py /D2/outputs/devtest_output /D5/outputs/sumbasic_summaries_no_stopwords -ry -1 prob 5
python ./sumbasic.py /D2/outputs/devtest_output /D5/outputs/sumbasic_summaries_stopwords -rn -1 prob 5

# lexrank
python ./lexrank.py /D2/outputs/devtest_output /D5/outputs/lexrank_compressed_devtest lexrank
python ./lexrank.py /D2/outputs/evaltest_output /D5/outputs/lexrank_compressed_evaltest lexrank
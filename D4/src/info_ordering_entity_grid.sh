#!/bin/sh

# LLR
python ./info_ordering_entity_grid.py /D4/data/back_corpus_file /D2/outputs/devtest_output /D4/outputs/LLR_summaries_no_stopwords_0.05 0.05 100 0 5 0.8
python ./info_ordering_entity_grid.py /D4/data/back_corpus_file /D2/outputs/devtest_output /D4/outputs/LLR_summaries_stopwords_0.05 0.05 100 1 5 0.8

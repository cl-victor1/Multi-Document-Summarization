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

# evaluations
python3 ./rouge_eval.py /D5/data/model_devtest /D5/outputs/D5_devtest/LLR_devtest > rouge_scores_LLR_devtest.out
python3 ./rouge_eval.py /D5/data/model_devtest /D5/outputs/D5_devtest/lexrank_compressed_devtest > rouge_scores_lexrank_devtest.out
python3 ./rouge_eval.py /D5/data/model_devtest /D5/outputs/D5_devtest/sumbasic_devtest > rouge_scores_sumbasic_devtest.out
python3 ./rouge_eval.py /D5/data/model_devtest /D5/data/TAC_sharedtask_devset > rouge_scores_peers_devtest.out
python3 ./rouge_eval.py /D5/data/model_evaltest /D5/outputs/D5_evaltest/LLR_evaltest > rouge_scores_LLR_evaltest.out
python3 ./rouge_eval.py /D5/data/model_evaltest /D5/outputs/D5_evaltest/lexrank_compressed_evaltest > rouge_scores_lexrank_evaltest.out
python3 ./rouge_eval.py /D5/data/model_evaltest /D5/outputs/D5_evaltest/sumbasic_evaltest > rouge_scores_sumbasic_evaltest.out

python ./infolm_eval.py ../data/model_devtest/ ../outputs/D5_devtest/LLR_devtest/ 1 infolm_output_LLR_devtest.csv
python ./infolm_eval.py ../data/model_devtest/ ../outputs/D5_devtest/lexrank_compressed_devtest/ 2 infolm_output_sumbasic_devtest.csv
python ./infolm_eval.py ../data/model_devtest/ ../outputs/D5_devtest/sumbasic_devtest/ 3 infolm_output_LexRank_devtest.csv
python ./pearson.py ../data/model_devtest/ ../outputs/D5_devtest/LLR_devtest/ 1 pearson_output_LLR_devtest.csv
python ./pearson.py ../data/model_devtest/ ../outputs/D5_devtest/sumbasic_devtest/ 2 pearson_output_sumbasic_devtest.csv
python ./pearson.py ../data/model_devtest/ ../outputs/D5_devtest/lexrank_compressed_devtest/ 3 pearson_output_lexrank_devtest.csv

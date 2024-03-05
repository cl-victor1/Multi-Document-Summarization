#!/bin/sh

# baseline
python ./infolm_eval.py ../data/model_devtest/ ../outputs/D5_devtest/LLR_devtest/ 1 infolm_output_LLR_devtest.csv
python ./infolm_eval.py ../data/model_devtest/ ../outputs/D5_devtest/lexrank_compressed_devtest/ 2 infolm_output_sumbasic_devtest.csv
python ./infolm_eval.py ../data/model_devtest/ ../outputs/D5_devtest/sumbasic_devtest/ 3 infolm_output_LexRank_devtest.csv
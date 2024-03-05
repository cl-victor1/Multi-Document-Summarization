#!/bin/sh

# baseline
python ./pearson.py ../data/model_devtest/ ../outputs/D5_devtest/LLR_devtest/ 1 pearson_output_LLR_devtest.csv
python ./pearson.py ../data/model_devtest/ ../outputs/D5_devtest/sumbasic_devtest/ 2 pearson_output_sumbasic_devtest.csv
python ./pearson.py ../data/model_devtest/ ../outputs/D5_devtest/lexrank_compressed_devtest/ 3 pearson_output_lexrank_devtest.csv
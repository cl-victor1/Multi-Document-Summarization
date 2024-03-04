#!/bin/sh

python ./sumbasic.py /D2/outputs/devtest_output /D4/outputs/sumbasic_summaries_no_stopwords -ry -1 prob 5
python ./sumbasic.py /D2/outputs/devtest_output /D4/outputs/sumbasic_summaries_stopwords -rn -1 prob 5

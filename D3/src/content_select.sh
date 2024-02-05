#!/bin/sh
pip install rouge-score
# make background corpus for LLR
/nopt/dropbox/23-24/570/envs/570/bin/python ./src/make_LLR_background.py $1 $2
# with and without stopwords
/nopt/dropbox/23-24/570/envs/570/bin/python ./src/LLR.py $1 $3 $4 $8 $9 $5
/nopt/dropbox/23-24/570/envs/570/bin/python ./src/LLR.py $1 $3 $6 $8 $9 $7

/nopt/dropbox/23-24/570/envs/570/bin/python ./src/sumBasic.py ??

#/nopt/dropbox/23-24/570/envs/570/bin/python ./src/lexrank.py $input $output $summary_length

#/dropbox/18-19/573/Data/models/devtest/
/nopt/dropbox/23-24/570/envs/570/bin/python ./src/evaluate.py $15 $16
/nopt/dropbox/23-24/570/envs/570/bin/python ./src/evaluate.py $15 $16
/nopt/dropbox/23-24/570/envs/570/bin/python ./src/evaluate.py $15 $16
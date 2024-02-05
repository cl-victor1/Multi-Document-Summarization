#!/bin/sh
pip install rouge-score
# $1 $2 are 
/nopt/dropbox/23-24/570/envs/570/bin/python ./src/make_LLR_background.py $1 $2

/nopt/dropbox/23-24/570/envs/570/bin/python ./src/LLR.py $1 $3 $4 $5 $6 $7

/nopt/dropbox/23-24/570/envs/570/bin/python ./src/sumBasic.py ??

/nopt/dropbox/23-24/570/envs/570/bin/python ./src/lexrank.py ??
#/dropbox/18-19/573/Data/models/devtest/
/nopt/dropbox/23-24/570/envs/570/bin/python ./src/evaluate.py $15 $16
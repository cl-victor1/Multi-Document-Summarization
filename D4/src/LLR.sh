#!/bin/sh

/nopt/dropbox/23-24/570/envs/570/bin/python ./make_LLR_background.py $1 $2
/nopt/dropbox/23-24/570/envs/570/bin/python ./LLR.py $1 $3 $4 $5 $6 $7
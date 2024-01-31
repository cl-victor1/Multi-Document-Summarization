#!/bin/sh

/nopt/dropbox/23-24/570/envs/570/bin/python ./extract.py $1
/nopt/dropbox/23-24/570/envs/570/bin/python ./extract.py $2
/nopt/dropbox/23-24/570/envs/570/bin/python ./extract.py $3

/nopt/dropbox/23-24/570/envs/570/bin/python ./training_process.py $1 $4 $5
/nopt/dropbox/23-24/570/envs/570/bin/python ./dev_process.py $2 $6 $7
/nopt/dropbox/23-24/570/envs/570/bin/python ./eval_process.py $3 $8 $9


#!/bin/sh

/nopt/dropbox/23-24/570/envs/570/bin/python ./extract.py $1
/nopt/dropbox/23-24/570/envs/570/bin/python ./extract.py $2
/nopt/dropbox/23-24/570/envs/570/bin/python ./extract.py $3

/nopt/dropbox/23-24/570/envs/570/bin/python ./process.py $@

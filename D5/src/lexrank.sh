#!/bin/sh

python ./lexrank.py ../../D2/outputs/devtest_output ../outputs/lexrank_compressed_devtest lexrank

python ./lexrank.py ../../D2/outputs/evaltest_output ../outputs/lexrank_compressed_evaltest lexrank

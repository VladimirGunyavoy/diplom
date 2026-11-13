#!/bin/bash
# usage: run6.sh ENV N SEED  (aida2, d4)
cd ~/spore_v5/d4
env ENVN=$1 NB=$2 SEED=$3 KLP=${KLP:-28} NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=${NT:-4} KDW=4 /usr/bin/time -v ~/spore_v5/venv/bin/python g6d.py > log6_$1_$2_$3.log 2> time6_$1_$2_$3.txt

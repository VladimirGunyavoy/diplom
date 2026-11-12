#!/bin/bash
# usage: run6.sh ENV N SEED   (не запускать без слова research-39!) ; m8: KLP=36 (K28 для 6D; для 8D 2d+... — как c12 для ДИ8D уточнить)
cd ~/spore_v5/d3; R=~/spore_v5/a38/spores_2/bench/b22/runpc.sh
env ENVN=$1 NB=$2 SEED=$3 KLP=${KLP:-28} NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=${NT:-4} KDW=4 /usr/bin/time -v $R g6d.py > log6_$1_$2_$3.log 2> time6_$1_$2_$3.txt

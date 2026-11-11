#!/bin/bash
# wv.sh TAG "ENV.." seed  [NB]
TAG=$1; X=$2; k=$3; cd ~/spore_v5/a37/spores_2/bench; PY=$HOME/spore_v5/r31/venv/bin/python
env EULER=1 QUANT=1 CLRDT=.05 PAD=.05 RHOB=.3 RHOF=.15 GMARG=.05 NUMBA1=1 OMP_NUM_THREADS=4 NREF=1 CEIL=1 NSTARTS=100 HARD=1 STARTSEED=${STARTSEED:-1} $X SEED=$k $PY w38_V.py ${NB:-6000} ${TAG}s$k > ~/spore_v5/a37/${TAG}s$k.log 2>&1

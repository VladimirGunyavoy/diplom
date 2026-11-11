#!/bin/bash
# w37: дрон R20-постановка, своя копия a37. dr.sh TAG "ENV=.. ENV=.." seed0 seed1 ...  (STARTSEED по умолч. 1 — подбор)
TAG=$1; X=$2; shift 2; cd ~/spore_v5/a37/spores_2/bench; PY=$HOME/spore_v5/r31/venv/bin/python
for k in "$@"; do
env EULER=1 QUANT=1 CLRDT=.05 PAD=.05 RHOB=.3 RHOF=.15 GMARG=.05 PROCS=8 NUMBA1=1 OMP_NUM_THREADS=1 NREF=1 CEIL=1 FBNF=6400 FBK=5 RETRY=3 NSTARTS=100 HARD=1 STARTSEED=1 $X SEED=$k $PY spores_drone_e.py 6000 ${NFF:-1600} ${KK:-5} _${TAG}s$k > ~/spore_v5/a37/${TAG}s$k.log 2>&1; echo "$(date +%T) ${TAG} seed $k done" >> ~/spore_v5/a37/dr.log; done

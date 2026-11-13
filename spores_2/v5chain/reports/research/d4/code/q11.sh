#!/bin/bash
cd ~/spore_v5/d4/spores_2/w39n
export NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=4 KDW=4 OMP_NUM_THREADS=1 RNDN=1 KLP=20 VCAP=40 KVAUTO=1 KV=.05 DEP=2 DEPPEN=1 TAUS=.2,.4,.8 TAU=.4 RB=.03 SKIPBASE=1 BOX=full WMARG=.05 WL=1 PROCS=4
r() { nice -n 10 ~/spore_v5/venv/bin/python refine_sl2.py $1 100000 8 $2 > ../../rsl2_$1_$2.log 2>&1; }
r dp1 0 & r manip 0 & wait
r dp1 1 & r manip 1 & wait
r dp1 2 & r manip 2 & wait

#!/bin/bash
# usage: full.sh ENV NB SEED — d4: полная коробка, nice, /usr/bin/time -v
cd ~/spore_v5/d4/spores_2/w39n
export NUMBA_THREADING_LAYER=workqueue OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4 KDW=4 NPROC=${NPROC:-4} RNDN=1 KLP=20 VCAP=40 KVAUTO=1 KV=.05 DEP=2 DEPPEN=1 TAUS=.2,.4,.8 TAU=.4 RB=.03 SKIPBASE=1 TRACE=0 BOX=full TAG=_full
exec nice -n 10 /usr/bin/time -v -o ../../out/vtime_$1_$2_$3.txt ~/spore_v5/venv/bin/python tvg_lp.py $1 $2 8 $3 > ../../out/v_$1_$2_$3.log 2>&1

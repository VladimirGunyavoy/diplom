#!/bin/bash
# охват радиусом: BUILDONLY-сборки maze4 с разными параметрами роста → квантили r клеток. aida: bash cover_m4.sh TAG KEY=VAL ...
TAG=$1; shift; cd ~/spore_v5/c1/spores_2/bench
env "$@" MAXC=${MAXC:-600} BUILDONLY=1 STGPU=0 TAG=_$TAG DUMPL=~/spore_v5/c1/m4c_$TAG.pkl python3 spores_maze4.py > ~/spore_v5/c1/logs/m4c_$TAG.log 2>&1
cd ~/spore_v5/c1/maze4_src && SYS=maze4 python3 - <<PY >> ~/spore_v5/c1/logs/m4c_$TAG.log 2>&1
import sys, pickle, numpy as np
sys.path.insert(0, "."); import growN
d = pickle.load(open("/home/random/spore_v5/c1/m4c_$TAG.pkl","rb"))["layers"]
C = np.array([np.asarray(c.c,float) for c in d[0]]); r = np.linalg.norm(C[:,:2],axis=1)
print("COVER $TAG", "n", len(C), "r q10/50/90/max", np.round(np.percentile(r,[10,50,90,100]),2), "r_cell med", np.round(np.median([np.mean(np.asarray(c.r)) for c in d[0]]),3))
PY

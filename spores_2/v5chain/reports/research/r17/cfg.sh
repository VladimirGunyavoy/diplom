#!/bin/bash
# r17: grow3 RS3 + finish VF1.5 + T/refbox; usage: cfg.sh tag ENV=... ...
tag=$1; shift; cd ~/spore_v5/r17/v7
env "$@" RS=3 RMIN=.04 MINROWS=5 NFAIL=60 DUMP=../out/atlas_$tag.pkl python3 src/cells7/grow3.py > ../out/$tag.log 2>&1
env "$@" RS=3 VF=1.5 python3 ../diag/finish.py ../out/atlas_$tag.pkl >> ../out/$tag.log 2>&1
python3 ../diag/ovl.py ../out/atlas_$tag.pkl >> ../out/$tag.log 2>&1
python3 -c "
import numpy as np; rb=np.load(\"../out/dd_refbox_60.npy\"); T=np.load(\"../out/atlas_$tag\"+\"_fin1.5.npy\"); m=np.isfinite(T); r=T[m]/rb[m]
print(\"RESULT $tag reach %d/60 T/refbox mean %.4f med %.4f max %.3f\" % (m.sum(), r.mean(), np.median(r), r.max()))" >> ../out/$tag.log 2>&1

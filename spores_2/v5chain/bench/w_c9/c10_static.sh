#!/bin/bash
cd $HOME/spore_v5/b17/spores_2/bench/b22
export TIEP=.5 ESCN=5 ESCD=5 ESCB=200 KV=.05 RB=1.5 TIE=.1 DEP=3 NSEL=32 
for nb in 50000 100000 200000; do bash runpc.sh car_val_c10.py $nb 8 sel 20 23 > c10_static_$nb.log 2>&1; mv car_val_sel_${nb}_0.05_1.5_20.json c10_static_$nb.json 2>/dev/null; done

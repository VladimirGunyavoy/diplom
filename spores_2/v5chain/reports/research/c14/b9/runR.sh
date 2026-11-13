cd ~/spore_v5/c14/m8/spores_2/bench
export CAREF=1 CAMS=20 TREEFAST=1 NEIGH=1 NFRETRY=40000 PROCS=8 FASTREPLAY=1 VECTREE=64
D=$HOME/spore_v5/c14/m8
R="nice -n 10 bash $HOME/spore_v5/a38/spores_2/bench/b22/runpc.sh"
CAIT=50 CUTINF=1 TRYLOG=$D/M8.tries.jsonl /usr/bin/time -v -o $D/M8.time $R spores_8d_bs.py m8 60000 15000 3 c14cut wlim > $D/M8.log 2>&1
NEIGH=0 CAIT=50 CUTINF=1 /usr/bin/time -v -o $D/M9.time $R spores_8d_bs.py m8 60000 15000 3 c14cutn0 wlim > $D/M9.log 2>&1

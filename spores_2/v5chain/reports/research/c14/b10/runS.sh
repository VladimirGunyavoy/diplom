cd ~/spore_v5/c14/m8/spores_2/bench
export CAREF=1 CAMS=20 TREEFAST=1 NEIGH=1 NFRETRY=40000 PROCS=8 FASTREPLAY=1 VECTREE=64 CAIT=50 CUTINF=1
D=$HOME/spore_v5/c14/m8
R="nice -n 10 bash $HOME/spore_v5/a38/spores_2/bench/b22/runpc.sh"
POLISH=P1 TRYLOG=$D/M10.tries.jsonl /usr/bin/time -v -o $D/M10.time $R spores_8d_bs.py m8 60000 15000 3 c14p1 wlim > $D/M10.log 2>&1
POLISH=P2 TRYLOG=$D/M11.tries.jsonl /usr/bin/time -v -o $D/M11.time $R spores_8d_bs.py m8 60000 15000 3 c14p2 wlim > $D/M11.log 2>&1

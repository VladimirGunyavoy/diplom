cd ~/spore_v5/c14/m8/spores_2/bench
export CAREF=1 CAMS=20 TREEFAST=1 NEIGH=1 NFRETRY=40000 PROCS=8
R="nice -n 10 bash $HOME/spore_v5/a38/spores_2/bench/b22/runpc.sh"
D=$HOME/spore_v5/c14/m8
TRYLOG=$D/M0.tries.jsonl /usr/bin/time -v -o $D/M0.time $R spores_8d_bs.py m8 60000 15000 3 c14ctl wlim > $D/M0.log 2>&1
FASTREPLAY=1 TRYLOG=$D/M1.tries.jsonl /usr/bin/time -v -o $D/M1.time $R spores_8d_bs.py m8 60000 15000 3 c14fast wlim > $D/M1.log 2>&1
FASTREPLAY=1 KP=20 GT=3 MACC=1 TRYLOG=$D/M2.tries.jsonl /usr/bin/time -v -o $D/M2.time $R spores_8d_loc.py m8 60000 15000 3 c14loc wlim > $D/M2.log 2>&1

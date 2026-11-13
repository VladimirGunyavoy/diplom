cd ~/spore_v5/c14/m8/spores_2/bench
export CAREF=1 CAMS=20 TREEFAST=1 NEIGH=1 NFRETRY=40000 PROCS=8 FASTREPLAY=1
R="nice -n 10 bash $HOME/spore_v5/a38/spores_2/bench/b22/runpc.sh"
D=$HOME/spore_v5/c14/m8
EARLYSTOP=1 TRYLOG=$D/M3.tries.jsonl /usr/bin/time -v -o $D/M3.time $R spores_8d_bs.py m8 60000 15000 3 c14es wlim > $D/M3.log 2>&1
VECTREE=64 /usr/bin/time -v -o $D/M4.time $R spores_8d_bs.py m8 60000 15000 3 c14vt wlim > $D/M4.log 2>&1
EARLYSTOP=1 VECTREE=64 /usr/bin/time -v -o $D/M5.time $R spores_8d_bs.py m8 60000 15000 3 c14all wlim > $D/M5.log 2>&1

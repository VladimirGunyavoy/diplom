cd ~/spore_v5/c14/m8/spores_2/bench
export CAREF=1 CAMS=20 TREEFAST=1 NEIGH=1 NFRETRY=40000 PROCS=8 FASTREPLAY=1 VECTREE=64
D=$HOME/spore_v5/c14/m8
TRYLOG=$D/M7.tries.jsonl /usr/bin/time -v -o $D/M7.time nice -n 10 bash $HOME/spore_v5/a38/spores_2/bench/b22/runpc.sh spores_8d_bs.py m8 60000 15000 3 c14b9 wlim > $D/M7.log 2>&1

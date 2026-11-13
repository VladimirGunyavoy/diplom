#!/bin/bash
# e2: блок слепой проверки (ii) на aida1: ≤4 процесса, nice 10
cd ~/spore_v5/e2; export W43=$HOME/spore_v5/e2 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 PROCS=4 NB=12000 NF=3000 WL=1 NSW=64 WMARG=.05
PY=/home/random/spore_v5/r31/venv/bin/python; mkdir -p out refnew
echo "$(date +%T) старт OCP новые старты" > out/blind.log
cd refnew; cp ../ref_e2.py .; export PYTHONPATH=$HOME/spore_v5/r23/pylib NQ=20 SEEDN=777 WMP=3 G=1
seq 0 19 | xargs -P4 -I{} sh -c 'nice -n 10 python3 ref_e2.py 160 4 {} > out_{}.jsonl 2> err_{}.log'
cd ..; unset PYTHONPATH
python3 - <<'PY' >> out/blind.log
import json,numpy as np
rows=[]
for q in range(20):
    d=json.loads(open(f'refnew/out_{q}.jsonl').readline()); rows.append([*d['x'], d['T'] if d['T'] is not None else np.nan, ])
a=np.array(rows); np.save('dp1_ref_new20.npy',a); print('ref new: T',np.round(a[:,4],3).tolist(),'finite',int(np.isfinite(a[:,4]).sum()))
PY
echo "$(date +%T) старт manip (ii)" >> out/blind.log
nice -n 10 $PY corridor_env.py manip _blind_ii > out/corr_manip_ii.log 2>&1
echo "$(date +%T) старт dp1 новые 20 (ii)" >> out/blind.log
STARTSNPY=$HOME/spore_v5/e2/dp1_ref_new20.npy nice -n 10 $PY corridor_env.py dp1 _new20_ii > out/corr_dp1_new20_ii.log 2>&1
echo "$(date +%T) старт профиль" >> out/blind.log
PROCS=1 nice -n 10 $PY profile_dp1.py 17 $HOME/spore_v5/e2/out/back_dp1_sl2.pkl > out/prof_dp1_17.txt 2>&1
echo ALLDONE > out/blind_done.txt

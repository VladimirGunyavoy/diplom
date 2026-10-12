#!/bin/bash
# C3/C5 на среде $1 (aida): свип SST на 5 стартах → лучший prune → SST на 20 + RRT + OCP. kRRT* — только ДИ.
cd ~/spore_v5/c1/spores_2/bench; P=~/venv-ompl/bin/python; E=$1; NP=${2:-8}
for pr in .05 .1 .2 .4; do $P run_sst.py $E --budgets 1,3,10 --starts 0,5,9,14,18 --prune $pr --procs 5 --tag _sw$pr > /dev/null 2>&1; done
BEST=$($P - <<PY
import json,glob,os
sys=__import__('sys'); best=None
for pr in ('.05','.1','.2','.4'):
    d=json.load(open('../../reports/bench/data/${E}_sst_sw%s.json'%pr)); b=d['budgets'][-1]; sc=(-b['reached'], b['ratio_med'] if b['ratio_med'] else 9)
    if best is None or sc<best[0]: best=(sc,pr)
print(best[1])
PY
)
echo "BEST prune $BEST"
$P run_sst.py $E --budgets .3,1,3,10,60 --prune $BEST --procs $NP
$P run_sst.py $E --algo rrt --budgets 60 --procs $NP
$P ocp_run.py $E --procs $NP
echo ENVDONE $E

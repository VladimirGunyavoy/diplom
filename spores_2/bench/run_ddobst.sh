#!/bin/bash
# CA1-1: дд + 2 диска. После OCPQDONE: споры (тихая машина) → SST/RRT/kRRT* + PPO. Лог ~/spore_v5/c1/ddobst.log
until grep -q OCPQDONE ~/spore_v5/c1/ocpq.log 2>/dev/null; do sleep 20; done
cd ~/spore_v5/c1/spores_2/bench; P=~/venv-ompl/bin/python; V=~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python
BENCH_ENV=ddobst python3 spores_dd.py > ~/spore_v5/c1/ddobst_spores.log 2>&1; echo SPORESDONE; tail -n1 ~/spore_v5/c1/ddobst_spores.log | cut -c1-300
for st in 1000000 5000000; do for sd in 0 1 2; do nohup $V ppo_run.py ddobst $st $sd > ~/spore_v5/c1/ppo_ddobst_${st}_s$sd.log 2>&1 & done; done
$P run_sst.py ddobst --algo rrtstar --procs 10 2>&1 | tail -1 | cut -c1-250
$P run_sst.py ddobst --algo rrt --budgets 60 --procs 10 2>&1 | tail -1 | cut -c1-250
$P run_sst.py ddobst --algo sst --prune .05 --procs 10 2>&1 | tail -1 | cut -c1-250
wait; echo DDOBSTDONE

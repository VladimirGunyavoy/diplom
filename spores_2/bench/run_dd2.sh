#!/bin/bash
# дд: kRRT* (Balkcom–Mason TGT), чувствительность к цели ±.2 (SST, RRT, PPO 5M). Лог ~/spore_v5/c1/dd2.log
cd ~/spore_v5/c1/spores_2/bench; P=~/venv-ompl/bin/python; V=~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python
PPO_GAMMA=.99 nohup $V ppo_run.py dd2 5000000 0 > ~/spore_v5/c1/ppo_dd2.log 2>&1 &
$P run_sst.py dd --algo rrtstar --procs 14 2>&1 | tail -3 | cut -c1-250
$P run_sst.py dd2 --algo sst --prune .05 --budgets 1,10,60 --procs 14 2>&1 | tail -1 | cut -c1-250
$P run_sst.py dd2 --algo rrt --budgets 60 --procs 14 2>&1 | tail -1 | cut -c1-250
wait; echo DD2DONE

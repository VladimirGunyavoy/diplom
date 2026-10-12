#!/bin/bash
cd ~/spore_v5/c1/spores_2/bench; P=~/venv-ompl/bin/python
while pgrep -f "[r]un_sst.py dd2 --algo sst" >/dev/null; do sleep 10; done
$P run_sst.py dd2 --algo rrt --budgets 60 --procs 14 2>&1 | tail -1 | cut -c1-250
$P run_sst.py dd --algo rrtstar --procs 14 2>&1 | tail -3 | cut -c1-250
echo DD3DONE

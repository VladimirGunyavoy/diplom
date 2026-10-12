#!/bin/bash
# C3 на ДИ: свип SST на 5 стартах, потом все 20 (aida). Лог ../../c3_di.log
cd ~/spore_v5/c1/spores_2/bench; P=~/venv-ompl/bin/python
for pr in .05 .1 .2 .4; do $P run_sst.py di --budgets 1,3,10 --starts 0,5,9,14,18 --prune $pr --procs 5 --tag _sw$pr; done
$P run_sst.py di --algo rrt --budgets 60 --procs 16
$P run_sst.py di --algo rrtstar --budgets .3,1,3,10,60 --procs 16
echo ALLDONE

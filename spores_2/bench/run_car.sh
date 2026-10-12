#!/bin/bash
# CA2 статик кар: после DDOBSTDONE — споры (коридор v6), потом SST/RRT/kRRT*(нет: без аналитического планировщика)/PPO. Лог ~/spore_v5/c1/car.log
until grep -q DDOBSTDONE ~/spore_v5/c1/ddobst.log 2>/dev/null; do sleep 20; done
cd ~/spore_v5/r21/v6; CARL=6 CM=.002 python3 ~/spore_v5/c1/spores_2/bench/spores_car.py > ~/spore_v5/c1/car_spores.log 2>&1; echo CARSPORESDONE; tail -n1 ~/spore_v5/c1/car_spores.log | cut -c1-300
cd ~/spore_v5/c1/spores_2/bench; P=~/venv-ompl/bin/python; V=~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python
for st in 1000000 5000000; do for sd in 0 1 2; do nohup $V ppo_run.py car $st $sd > ~/spore_v5/c1/ppo_car_${st}_s$sd.log 2>&1 & done; done
$P run_sst.py car --algo rrt --budgets 60 --procs 10 2>&1 | tail -1 | cut -c1-250
$P run_sst.py car --algo sst --prune .1 --procs 10 2>&1 | tail -1 | cut -c1-250
wait; echo CARDONE

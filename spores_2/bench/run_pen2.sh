#!/bin/bash
# PPO со штрафом выхода −25, γ .995, 5M × 3 сида на ddobst и car (после находки «самоубийства»). Лог ~/spore_v5/c1/pen2.log
cd ~/spore_v5/c1/spores_2/bench; V=~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python
$V ppo_weight.py 2>&1 | tail -1 | cut -c1-100
for e in ddobst car; do for sd in 0 1 2; do PPO_BOXPEN=25 PPO_GAMMA=.995 nohup $V ppo_run.py $e 5000000 $sd > ~/spore_v5/c1/ppo_${e}_pen_s$sd.log 2>&1 & done; done
wait; echo PEN2DONE

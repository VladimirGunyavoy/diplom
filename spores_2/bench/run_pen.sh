#!/bin/bash
# PPO со штрафом выхода −(TMAXEP+5): γ .995 и .99, pend и manip, 5M, сид 0. Лог ~/spore_v5/c1/pen.log
cd ~/spore_v5/c1/spores_2/bench; V=~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python
for g in .995 .99; do for e in pend manip; do PPO_BOXPEN=25 PPO_GAMMA=$g nohup $V ppo_run.py $e 5000000 0 > ~/spore_v5/c1/ppo_${e}_pen_g$g.log 2>&1 & done; done
wait; echo PENDONE

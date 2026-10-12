#!/bin/bash
# после QUIETDONE: OCP CasADi (pend dd manip), затем PPO γ .995/.999 (pend, manip, 5M, сид 0). Лог ~/spore_v5/c1/followup.log
until grep -q QUIETDONE ~/spore_v5/c1/quiet.log 2>/dev/null; do sleep 20; done
cd ~/spore_v5/c1/spores_2/bench
for e in pend dd manip; do ~/venv-ompl/bin/python ocp_casadi.py $e --procs 20 2>&1 | tail -1 | cut -c1-400; done
echo OCPCDONE
V=~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python
for g in .995 .999; do for e in pend manip; do PPO_GAMMA=$g nohup $V ppo_run.py $e 5000000 0 > ~/spore_v5/c1/ppo_${e}_g${g}.log 2>&1 & done; done
wait; echo PPOGDONE

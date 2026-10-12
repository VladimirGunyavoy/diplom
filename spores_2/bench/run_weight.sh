#!/bin/bash
# после ocpc: перезапись спор (тихая машина) с полем weight. Лог ~/spore_v5/c1/weight.log
while pgrep -f "[o]cp_casadi.py" >/dev/null; do sleep 15; done
cd ~/spore_v5/c1/spores_2/bench
OMP_NUM_THREADS=1 M=5 python3 spores_di.py 2>&1 | tail -1 | cut -c1-200
for v in 84 56; do VARIANT=$v python3 spores_pend.py 2>&1 | tail -1 | cut -c1-200; done
python3 spores_dd.py 2>&1 | tail -1 | cut -c1-200
MODE=load python3 spores_manip.py 2>&1 | tail -1 | cut -c1-200
echo WEIGHTDONE

#!/bin/bash
# спокойный тайминг споров (после PPO и manip_full): ДИ, маятник 84/56, дд — тех же скриптов, пишет в те же data/*.json (перезапись). Лог ~/spore_v5/c1/quiet.log
until grep -q MANIPFULLDONE ~/spore_v5/c1/manip_full.log 2>/dev/null; do sleep 20; done
cd ~/spore_v5/c1/spores_2/bench
OMP_NUM_THREADS=1 M=5 python3 spores_di.py 2>&1 | tail -1 | cut -c1-300
for v in 84 56; do VARIANT=$v python3 spores_pend.py 2>&1 | tail -1 | cut -c1-300; done
python3 spores_dd.py 2>&1 | tail -1 | cut -c1-300
echo QUIETDONE

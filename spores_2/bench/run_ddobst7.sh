#!/bin/bash
# подмножество ddobst7 (BM-TGT пересекает диск): споры (пересборка атласа) + kRRT*-BM. Лог ~/spore_v5/c1/ddobst7.log
cd ~/spore_v5/c1/spores_2/bench; P=~/venv-ompl/bin/python
$P run_sst.py ddobst7 --algo rrtstar --procs 7 2>&1 | tail -1 | cut -c1-250
BENCH_ENV=ddobst7 python3 spores_dd.py > ~/spore_v5/c1/ddobst7_spores.log 2>&1; tail -n1 ~/spore_v5/c1/ddobst7_spores.log | cut -c1-300
echo DD7DONE

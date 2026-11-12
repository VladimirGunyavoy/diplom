#!/bin/bash
# w43 [A] (PLAN ★★★ 06:45): m3gdg, 5 зёрен обоих вариантов + lookahead/тупики. ЗАПУСКАТЬ ТОЛЬКО ПО СЛОВУ research-39 «можно» (машины согласованы с journal-2026).
# На aida2: free -g, запас ≥10 ГБ. Порядок: A) равномерный 200k (зёрна 0-4, DEP=1) ; B) dense 150k+50k (зёрна 0-4); C) тест тупиков: 200k DEP=2 VIAB=1 зерно 0..2.
cd ~/spore_v5/w43; export NUMBA_NUM_THREADS=8 KDW=8 PYTHONPATH=$PWD/spores_2/bench:$PWD/spores_2/v6
PY=${PY:-~/spore_v5/venv/bin/python}
for s in 0 1 2 3 4; do NB=200000 KLP=20 VCAP=40 TAUS=.1,.2,.4 TAG=_stop nohup $PY m3gdg_graph.py 19 $s > out/g200k_stop_s$s.log 2>&1 & done; wait
for s in 0 1 2 3 4; do NU=150000 NT=50000 TT=.3 TAG=_stop nohup $PY m3gdg_dense.py 19 $s > out/dense_stop_s$s.log 2>&1 & done; wait
for s in 0 1 2; do DEP=2 VIAB=1 NB=200000 KLP=20 VCAP=40 TAUS=.1,.2,.4 TAG=_viab nohup $PY m3gdg_graph.py 19 $s > out/g200k_viab_s$s.log 2>&1 & done; wait

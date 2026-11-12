#!/bin/bash
# w43: очередь на aida1 по условиям research-39 (1 процесс, ≤4 потока, nice 10, free ≥10 ГБ запаса). Старт не раньше 07:45.
cd ~/spore_v5/w43; export NUMBA_NUM_THREADS=4 KDW=4 OMP_NUM_THREADS=1 PYTHONPATH=$PWD/spores_2/bench:$PWD/spores_2/v5chain
export PYTHONPATH=$PWD/spores_2/bench:$PWD/spores_2/v6
RUN="nice -n 10 bash $HOME/spore_v5/a38/spores_2/bench/b22/runpc.sh"
while [ $(date +%H%M) -lt 0745 ]; do sleep 30; done
run() { # имя, команда-окружение..., скрипт, зерно
  name=$1; shift
  while [ $(free -g | awk 'NR==2{print $7}') -lt 14 ]; do echo "$(date +%T) жду памяти" >> out/queue.log; sleep 60; done
  echo "$(date +%T) старт $name | $(uptime | sed 's/.*load/load/') | $(free -g | awk 'NR==2{print "avail "$7" ГБ"}')" >> out/queue.log
  env "$@" > out/$name.log 2>&1; echo "$(date +%T) конец $name: $(tail -2 out/$name.log | head -1 | cut -c1-200)" >> out/queue.log
}
for s in 0 1 2; do
  run g200k_stop_s$s NB=200000 KLP=20 VCAP=40 TAUS=.1,.2,.4 TAG=_stop $RUN m3gdg_graph.py 19 $s
  run g200k_viab_s$s DEP=2 VIAB=1 NB=200000 KLP=20 VCAP=40 TAUS=.1,.2,.4 TAG=_viab $RUN m3gdg_graph.py 19 $s
  run dense_stop_s$s NU=150000 NT=50000 TT=.3 TAG=_stop $RUN m3gdg_dense.py 19 $s
done
for s in 3 4; do
  run g200k_stop_s$s NB=200000 KLP=20 VCAP=40 TAUS=.1,.2,.4 TAG=_stop $RUN m3gdg_graph.py 19 $s
  run dense_stop_s$s NU=150000 NT=50000 TT=.3 TAG=_stop $RUN m3gdg_dense.py 19 $s
done

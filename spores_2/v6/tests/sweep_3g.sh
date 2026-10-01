#!/bin/bash
# (в) sweep 3 зв.+g вниз, 16 запросов, NB12000 NF3000; поток A: DT/KN, поток B: WH/TR (последовательно внутри потока)
cd ~/spore_v5/w7/v6; export OMP_NUM_THREADS=1 NQ=16 G=.3 DOWN=1
run() { n=$1; shift; env NAME=sw3_$n "$@" python3 tests/stats_manip6.py 12000 3000 ${TR:-3} > sw3_$n.log 2>&1; grep -ah "^NAME" sw3_$n.log | cut -c1-140; }
( run dt.05 DT=.05; run dt.02 DT=.02; run dt.1 DT=.1; run kn4 DT=.02 KN=4; run kn16 DT=.02 KN=16 ) > sw3A.out 2>&1 &
( run wh0 DT=.02 WH=0; run wh1.5 DT=.02 WH=1.5; run wh6 DT=.02 WH=6; TR=1 run tr1 DT=.02; TR=5 run tr5 DT=.02 ) > sw3B.out 2>&1 &
wait

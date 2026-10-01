#!/bin/bash
# (в) one-at-a-time sweep, 2 зв.+g вверх 4D, 40 запросов; базис DT=.05 KN=8 WH=3 TR=3 NB3000 NF2000
cd ~/spore_v5/w7/v6; export OMP_NUM_THREADS=1 NQ=40
run() { NAME=sw_$1 nohup python3 tests/stats_manip2g.py $2 $3 3 > sw_$1.log 2>&1; grep -ah "^NAME" sw_$1.log | cut -c1-120; }
for dt in .1 .05 .02; do DT=$dt run dt$dt 3000 2000; done
for kn in 2 4 8 16; do KN=$kn run kn$kn 3000 2000; done
for wh in 0 1.5 3 6; do WH=$wh run wh$wh 3000 2000; done
for tr in 1 2 3 5; do TR=$tr run tr$tr 3000 2000; done
for nb in "1000 500" "3000 2000" "6000 4000" "12000 8000"; do set -- $nb; run nb$1 $1 $2; done

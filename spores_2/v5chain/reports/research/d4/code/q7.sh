#!/bin/bash
# парно dp1: без/с WMARG .05 (граф тот же). fullv.sh = без, fullm.sh = с
cd ~/spore_v5/d4
run() { if [ "$1" = v ]; then ./fullv.sh $2 $3 $4; else WMARG=.05 ./fullm.sh $2 $3 $4; fi; }
export -f run
{
for s in 0 1 2; do echo "v dp1 100000 $s"; done
for s in 0 1 2 3; do echo "m dp1 100000 $s"; done
echo "v dp1 300000 0"; echo "m dp1 300000 0"; echo "m dp1 300000 1"
} | xargs -P 2 -L 1 bash -c 'run $0 $1 $2 $3'

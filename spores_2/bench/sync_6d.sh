#!/bin/bash
# линия A: bench + v6 + эталоны m3d на aida (~/spore_v5/a27/spores_2), результаты обратно: bash sync_6d.sh pull
cd "$(dirname "$0")/.."; A=aida:spore_v5/a27
if [ "$1" = pull ]; then mkdir -p ../reports/bench/data; rsync -a $A/reports/bench/data/ ../reports/bench/data/; exit; fi
ssh aida 'mkdir -p ~/spore_v5/a27/spores_2/v8 ~/spore_v5/a27/reports/bench/data/m3d_ref'
rsync -a --exclude __pycache__ bench $A/spores_2/; rsync -a --exclude __pycache__ v6 $A/spores_2/; rsync -a --exclude __pycache__ v8/src $A/spores_2/v8/
rsync -a ../reports/bench/data/m3d_ref/ $A/reports/bench/data/m3d_ref/

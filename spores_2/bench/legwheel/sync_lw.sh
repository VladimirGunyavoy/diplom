#!/bin/bash
# линия D: bench + v6 + v8/src + код робота (КОПИЯ чужого проекта, только чтение) и веса PPO → aida:~/spore_v5/d1; результаты обратно: bash sync_lw.sh pull
cd "$(dirname "$0")/../.."; A=aida:spore_v5/d1; CM=$HOME/claude-work/projects/colab-mhlb
if [ "$1" = pull ]; then mkdir -p ../reports/bench/data; rsync -a $A/reports/bench/data/ ../reports/bench/data/; exit; fi
ssh aida 'mkdir -p ~/spore_v5/d1/spores_2/v8 ~/spore_v5/d1/reports/bench/data ~/spore_v5/d1/colab/runs'
rsync -a --exclude __pycache__ bench $A/spores_2/; rsync -a --exclude __pycache__ v6 $A/spores_2/; rsync -a --exclude __pycache__ v8/src $A/spores_2/v8/
rsync -a --exclude __pycache__ --exclude '*.pkl*' $CM/legwheel_robot $A/colab/
rsync -a $CM/runs/legwheel_robot/2026-09-14/003_165713/ $A/colab/runs/003_165713/ --exclude live --exclude 'weights/00[0-6]*' --exclude 'evals/data/00[0-6]*'

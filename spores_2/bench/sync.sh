#!/bin/bash
# синхронизация bench + v8/src на aida (~/spore_v5/c1/spores_2); результаты — обратно: bench/pull.sh
cd "$(dirname "$0")/.."
ssh aida 'mkdir -p ~/spore_v5/c1/spores_2/v8 ~/spore_v5/c1/reports/bench/data'
rsync -a --exclude __pycache__ bench aida:~/spore_v5/c1/spores_2/
rsync -a --exclude __pycache__ v8/src aida:~/spore_v5/c1/spores_2/v8/
R=v5chain/reports/research
ssh aida 'mkdir -p ~/spore_v5/c1/spores_2/v5chain/reports/research/r23'
rsync -a $R/pend_ref_T.npy $R/pend_ref_best_u*.npy $R/dd_refbox_60.npy $R/dd_ref_60.npy aida:~/spore_v5/c1/spores_2/v5chain/reports/research/
rsync -a $R/r23/manip_ref_ocp_20.npy $R/r23/dp1_ref_ocp_20.npy aida:~/spore_v5/c1/spores_2/v5chain/reports/research/r23/

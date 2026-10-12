#!/bin/bash
# синхронизация bench + v8/src на aida (~/spore_v5/c1/spores_2); результаты — обратно: bench/pull.sh
cd "$(dirname "$0")/.."
ssh aida 'mkdir -p ~/spore_v5/c1/spores_2/v8 ~/spore_v5/c1/reports/bench/data'
rsync -a --exclude __pycache__ bench aida:~/spore_v5/c1/spores_2/
rsync -a --exclude __pycache__ v8/src aida:~/spore_v5/c1/spores_2/v8/

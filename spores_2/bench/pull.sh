#!/bin/bash
# результаты с aida → reports/bench/data
cd "$(dirname "$0")/../.."; mkdir -p reports/bench/data; rsync -a aida:~/spore_v5/c1/reports/bench/data/ reports/bench/data/

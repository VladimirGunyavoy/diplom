#!/bin/bash
# c9: дерево+V динамик кар TIGHT на aida 2. usage: run_c9.sh K0 K1 TAG [ONLY=0,1,..]
B=$HOME/spore_v5/c9; cd $B/spores_2/bench/b22; PY=$HOME/spore_v5/venv/bin/python
CU=$HOME/agentic-cad/data/tools/cuda-12.8
export CUDA_HOME=$CU NUMBA_CUDA_NVVM=$CU/nvvm/lib64/libnvvm.so NUMBA_CUDA_LIBDEVICE=$CU/nvvm/libdevice
export TIGHT=1 KV=.05 RB=1.5 FM=.05 TIE=.1 DEP=3 HYST=0 GPUVI=1 LIMV=.5 TOLV=1e-2 WARM=1 COMPACT=$B/r20_dyn_tight_compact.json
[ -n "$4" ] && export ONLY=$4
exec $PY car_dyn_c9.py 500000 8 test 0 $1 $2 > $B/$3.log 2>&1

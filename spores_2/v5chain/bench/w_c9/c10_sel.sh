#!/bin/bash
# usage: c10_sel.sh NB  (динамик sel, финальное правило)
NB=$1; B=$HOME/spore_v5/c9; cd $B/spores_2/bench/b22
PY=$HOME/spore_v5/venv/bin/python; CU=$HOME/agentic-cad/data/tools/cuda-12.8
export CUDA_HOME=$CU NUMBA_CUDA_NVVM=$CU/nvvm/lib64/libnvvm.so NUMBA_CUDA_LIBDEVICE=$CU/nvvm/libdevice
export TIEP=.5 ESCN=5 ESCD=5 ESCB=200 KV=.05 RB=1.5 FM=.05 TIE=.1 DEP=3 HYST=0 GPUVI=1 LIMV=.5 TOLV=1e-2 WARM=1 NSEL=32 VTAG=c10nb$NB
exec $PY car_dyn_c9c.py $NB 8 sel 20 20 23 > $B/c10_sel_$NB.log 2>&1

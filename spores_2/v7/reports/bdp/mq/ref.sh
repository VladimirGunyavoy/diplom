# $1 индекс старта, $2 K
I=$1; K=$2; R=~/spore_v5/w19; cd ~/spore_v5/r12
export PYTHONPATH=.:~/spore_v5/r5/pylib OMP_NUM_THREADS=1 X0=$(sed -n ${I}p $R/starts.txt) OUT=$R/ref_${I}_K$K.npy
TLO=${TLO:-2} THI=${THI:-9} python3 ocp_arcs.py $K 8 ${NR:-30} > $R/ref_${I}_K$K.out 2>&1

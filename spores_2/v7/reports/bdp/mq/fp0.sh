# w19 п.13: ход по рёбрам (FORCEPLAN=0) на атласе зерна $2-тега → refine → drop. $1 старт, $2 тег зерна ('' или b/c), $3 SD (зерно)
I=$1; TG=$2; SD=${3:-$I}; R=~/spore_v5/w19; cd ~/spore_v5/w17/v7
export PYTHONPATH=.:~/spore_v5/r5/pylib WCHK=1 WFILT=1 WNF=40 G=2 WIN=1.0 OMP_NUM_THREADS=2 START=$(sed -n ${I}p $R/starts.txt)
L=$(ls -t butterfly_dp_grow_N6000_tr1_fr1_fwd3000_R0.15_g2_rrt_kn4_nt_s${SD}_x${I}_lazy?.npz | head -1)
FORCEPLAN=0 TRAJALL=1 python3 reports/bdp/trajdump.py $L $R/tr_${I}${TG}f0.npz > $R/td_${I}${TG}f0.out 2>&1
python3 -m src.cells7.refine $R/tr_${I}${TG}f0.npz $R/rf_${I}${TG}f0.npz > $R/rf_${I}${TG}f0.out 2>&1
bash $R/drop.sh $R/rf_${I}${TG}f0.npz "$START" 6

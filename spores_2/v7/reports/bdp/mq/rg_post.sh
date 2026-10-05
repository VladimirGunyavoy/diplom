# w19 п.15: атлас после regrow (_rg.npz) → trajdump (FORCEPLAN=1) → refine → drop. $1 индекс старта
I=$1; R=~/spore_v5/w19; cd ~/spore_v5/w17/v7
export PYTHONPATH=.:~/spore_v5/r5/pylib WCHK=1 WFILT=1 WNF=40 G=2 WIN=1.0 OMP_NUM_THREADS=2 START=$(sed -n ${I}p $R/starts.txt)
A=butterfly_dp_grow_N6000_tr1_fr1_fwd3000_R0.15_g2_rrt_kn4_nt_s${I}_x${I}_rg.npz
FORCEPLAN=1 TRAJALL=1 python3 reports/bdp/trajdump.py $A $R/tr_${I}rg.npz > $R/td_${I}rg.out 2>&1
python3 -m src.cells7.refine $R/tr_${I}rg.npz $R/rf_${I}rg.npz > $R/rf_${I}rg.out 2>&1
bash $R/drop.sh $R/rf_${I}rg.npz "$START" 6

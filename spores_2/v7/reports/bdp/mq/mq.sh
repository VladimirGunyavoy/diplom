# w19 п.14: $1 индекс старта (строка starts.txt, с 1), $2 N (grow), $3 FWD. Запуск на aida из ~/spore_v5/w19 (код — w17/v7).
I=$1; N=${2:-6000}; F=${3:-3000}; R=~/spore_v5/w19; cd ~/spore_v5/w17/v7
export PYTHONPATH=.:~/spore_v5/r5/pylib WCHK=1 WFILT=1 WNF=40 G=2 WIN=1.0 NT=1 TR=1 FR=1 RRT=1 KN=4 KF=1.3 DMIN=.2 OMP_NUM_THREADS=2
export START=$(sed -n ${I}p $R/starts.txt) XTAG=$I SEED=$I FWD=$F
python3 -m src.cells7.butterfly_dp grow $N > $R/grow_$I.out 2>&1
A=butterfly_dp_grow_N${N}_tr1_fr1_fwd${F}_R0.15_g2_rrt_kn4_nt_s${I}_x$I
ROUNDS=6 CAP=4000 python3 -m src.cells7.butterfly_dp lazy $A.npz > $R/lazy_$I.out 2>&1
L=$(ls -t ${A}_lazy?.npz 2>/dev/null | head -1); [ -z "$L" ] && L=$A.npz
FORCEPLAN=1 TRAJALL=1 python3 reports/bdp/trajdump.py $L $R/tr_$I.npz > $R/td_$I.out 2>&1
python3 -m src.cells7.refine $R/tr_$I.npz $R/rf_$I.npz > $R/rf_$I.out 2>&1
bash $R/drop.sh $R/rf_$I.npz "$START" 6   # п.13а: топологии после доводки (результат — rf_$I_drop.npy/.out)

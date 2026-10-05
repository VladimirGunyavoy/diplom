# w19 п.13а: поиск топологий после доводки (порт research-12 dp_arc_drop INS=1 над rf_*.npz). $1 путь rf.npz (U,H), $2 X0 "q1,q2,w1,w2" (пусто = висит), $3 раундов
F=$1; X=${2:-}; R=${3:-4}; cd ~/spore_v5/r12
python3 -c "
import numpy as np,sys
d=np.load('$F'); np.save('${F%.npz}.npy', np.concatenate([[d['H'].sum()], d['U'].ravel(), d['H']]))"
export PYTHONPATH=.:~/spore_v5/r5/pylib G=2 INS=1 OMP_NUM_THREADS=1; [ -n "$X" ] && export X0=$X
python3 dp_arc_drop.py ${F%.npz}.npy $R > ${F%.npz}_drop.out 2>&1

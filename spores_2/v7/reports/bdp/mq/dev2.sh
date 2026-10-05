# w20 п.3: девиации пути по рёбрам (DEVS шаг, DEVR ранг) → trajdump → refine. $1 — старт i (с тем же зерном i, lazy1-атлас w17). Итог: dev_$1.res
I=$1; R=~/spore_v5/w19; D=~/spore_v5/w20; cd $D
export PYTHONPATH=.:~/spore_v5/r5/pylib WCHK=1 WFILT=1 WNF=40 G=2 WIN=1.0 NT=1 TR=1 FR=1 RRT=1 KN=4 KF=1.3 DMIN=.2 OMP_NUM_THREADS=2 TRAJALL=1 FORCEPLAN=0 DEVALL=1 TOPE=6
L=~/spore_v5/w17/v7/butterfly_dp_grow_N6000_tr1_fr1_fwd3000_R0.15_g2_rrt_kn4_nt_s${I}_x${I}_lazy1.npz
: > $D/dev2_$I.res
for S in -1 0 1 2 3 4 5 6 7 8 9 10 11 12; do for Rk in 1 2 3; do [ $S = -1 ] && [ $Rk != 1 ] && continue
  DEVS=$S DEVR=$Rk python3 reports/bdp/trajdump.py $L $D/dv_${I}_${S}_$Rk.npz > $D/dvd_${I}_${S}_$Rk.out 2>&1 || cp ~/spore_v5/w17/v7/reports/bdp/trajdump.py reports/bdp/ 2>/dev/null
  echo "S=$S R=$Rk traj: $(grep -h '"T"' $D/dvd_${I}_${S}_$Rk.out | cut -c1-120)" >> $D/dev2_$I.res
  python3 -m src.cells7.refine $D/dv_${I}_${S}_$Rk.npz $D/dvr_${I}_${S}_$Rk.npz 2>&1 | grep '"path"' | cut -c1-200 >> $D/dev2_$I.res
done; done

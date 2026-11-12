#!/bin/bash
# Одна строка таблицы (А): run_row.sh ENV NB SEED TAG [доп. env через переменные окружения]. НЕ ЗАПУСКАЛСЯ (машины возвращены, ждём «можно» research-39).
# Раскладка на машине: ~/spore_v5/tab/{tvg_vt,tvg_lp,tvg_dl}.py + spores_2/{bench,v6} (симлинки) + out/. Нарушения (viol_box/nogoal/slow) — в json/логе run_agent; память — time -v + sampler.sh (c12).
# По умолчанию: ДИ4D/6D/8D — TAUS .075,.15,.3, KLP=4d+4 (4D 20, 6D 28, 8D 36), manip/dp1 — BOX=full TAUS .2,.4,.8 KLP 20. free -g ПЕРЕД запуском: запас 10 ГБ (300k ~12 ГБ, 1M — до оценки по пику 300k).
ENV=$1; NB=$2; SEED=$3; TAG=$4; O=$HOME/spore_v5/tab/out; mkdir -p $O; cd $HOME/spore_v5/tab
case $ENV in di4) KLP=20; TAUS=.075,.15,.3; TAU=.15;; di6) KLP=28; TAUS=.075,.15,.3; TAU=.3; RB=.35;; di8) KLP=36; TAUS=.075,.15,.3; TAU=.3;; manip|dp1) KLP=20; TAUS=.2,.4,.8; TAU=.4; BOX=full;; esac
export NUMBA_THREADING_LAYER=workqueue OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=${NT:-8} KDW=${NT:-8} NPROC=${NPROC:-8} RNDN=1 KLP=$KLP VCAP=40 KVAUTO=1 KV=.05 DEP=2 DEPPEN=1 TAUS=$TAUS TAU=$TAU RB=${RB:-.03} SKIPBASE=1 TAG=_$TAG ${BOX:+BOX=$BOX} ${STARTS:+STARTS=$STARTS}
/usr/bin/time -v -o $O/$TAG.time ${PY:-python3} tvg_lp.py $ENV $NB 8 $SEED > $O/$TAG.log 2>&1

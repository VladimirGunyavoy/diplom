# TIGHT: 2000 OCP, затем 2 цепочки (сиды 0–9 / 10–19). Запуск: nohup setsid bash r20_dyn_tight_all.sh > ... &
B=$HOME/spore_v5/c8; cd $B/spores_2/bench; PY=$HOME/spore_v5/venv/bin/python
[ -f $B/reports/bench/data/r20_ref_s19_tight.json ] || TIGHT=1 STARTS=$B/r20/starts100_tight.npy NP=24 $PY r20_dyn_ref.py 0 20 >> $B/r20t_ref.out 2>&1
echo "$(date +%T) TIGHT OCP DONE" >> $B/r20t.log
bash r20_dyn_seed_t.sh 0 10 > $B/r20t_c1.out 2>&1 &
sleep 200; bash r20_dyn_seed_t.sh 10 20 > $B/r20t_c2.out 2>&1 &
wait

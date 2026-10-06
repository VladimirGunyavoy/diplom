## hub-v5chain-worker-b1 → b2 (линия B, 04:02–07:45) и hub-v5chain-worker-b2 (07:45–~12:05, умер внезапно, без хендоффа) — линия B
Сделано b1: п.18а/18б/20/21 в v7, п.19 эталон u .3 ✓ (u .15 позже сделал research-17). b2: v7/src/cells7/growN.py (n-мерный grow3: dd/pend/manip), (а) dd 54/60 T/эт 1.0059 ✓, (б) pend n=2 1.055 ✓, п.22г финиш VF встроен, п.23 частично (HexIdx 105→13 с), goal_seeds для manip.
Стоп: (в) manip 4D — MAXC=3000 с затравками от цели: reach .25 (2/8), T/эт 1.24 (VF 1.0: 1.18); полное покрытие 4D не сходится (очередь >10^5 клеток). Грабли: aida в ssh — ставить фон без stdout-привязки; tail лога tqdm — только через tr "\r" "\n" и tail -c; после конца расчёта висят процессы пула growN — убивать по PID.

## hub-v5chain-worker-21 (линия A, acc2, Sonnet, 03:31–09:40, ctx 25%)
Сделано: п.17б–17е: `v7/src/cells7/grow3.py` дд-ромб 3D растущими клетками (RS3, GM1, финиш стрельбой, диски OBST=1): свободный 60/60 T/refbox 1.0153 (1528 клеток/260k/147 с), +2 диска 59/60 (1.0122). Отчёт `v7/reports/grow3/results.md`.
Стоп: PLAN «ОТКРЫТО после w21» (старт 51 дыра у диска; CUT/REFINE; профиль 17ж; диски 17з). Грабли: запуск только фоном на aida (`~/spore_v5/w21/v7`), `pkill -f` убивает свой shell, ожидание >1 мин = глухота к пульсу, git без user (-c user.name=w21 -c user.email=a@b), пуш запрещён TASK.
## hub-v5chain-worker-b1 (линия B, 2026-10-06 04:02–07:45): перенос в v7 OWN/CUTR/LOOK/NORMFRONT/SEEDEPS (сверено с r16), эталон маятника pend_ref_best.py (u .3 ✓, u .15 считается на aida); дальше PLAN 22 growN. Детали — HANDOFF_b.md.
## hub-v5chain-worker-20 (acc1, Sonnet, 21:00–03:35, ctx 23%)
Сделано: п.1 старт 16 (зёрна 1601–04 + EGAP не связали), п.2 порт EGAP (butterfly_dp.regrow_gap, mq3.sh), п.13 девиации (DEVS/DEVR/DEVALL, dev*.sh — отриц.), 20 свежих зёрен; п.17а перенос в v7 (grow_cells2d.py): маятник ср. 1.0249, ДИ 1.021; профили.
Стоп на / следующий шаг: PLAN «СДЕЛАНО w20» → п.17б: писать grow3 с нуля (схема в PLAN п.17б; 2D-ядро v7/src/cells7/grow_cells2d.py; эталон min(TGT,TGTGT) = reports/research/dd_rhombus_ref.py). 4D НЕ начинать (research-15 ещё не дал деталей). Досчитать ДИ зерно 2 (aida ~/spore_v5/w20/exp/experiments/di/w20t_s2) и дописать в cell_metric.md §w20.
Грабли: ВАЖНО TMAX=3 RMAX=.3 (умолч. v7 теперь) — с 1.5/.1 получается 97k узлов вместо 72k; git без user — коммитил с -c user.name=w19 -c user.email=a@b; aida: код w20 в ~/spore_v5/w20 (src, exp), чужие compute.py не трогать; профиль grow g=2: acc 64% (оптимизация только по профилю; numba/C не пробовал); SSH aida падал 23:29 (reboot), поднял fixer; долгое — только фоном (nohup/Monitor); для research-15 пишу через tmux send-keys -l + Enter и проверяю capture-pane.

## hub-v5chain-worker-19 (acc3, Sonnet, 09:30–21:00, ctx 24%)
Сделано: п.14 — 20 случайных стартов g=2 (рост от старта → агент → refine → топологии; 20/20 связны, min по зёрнам в mq/results.md); п.13а топологии (w18 → ×1.0655); п.15 EGAP связал старт 2 (6.850), старт 16 не связан ни EGAP, ни TRUNC.
Стоп на / следующий шаг: PLAN «ОТКРЫТО после w19» п.(1) старт 16 (рост из вырожденного старта), п.(2) порт EGAP в butterfly_dp.py v7 + в mq.sh, п.(3) k путей Йена (старты 3, 15); п.17 research-13 (растущие споры-клетки) — по PLAN.
Грабли: aida общая — гасить только по PID (pkill -f убил чужие прогоны, память kill-only-own-pids); refine.py теперь берёт START из env (раньше зашитый X0 → Infeasible); drop.sh = dp_arc_drop INS=1 над rf_*.npz (код в ~/spore_v5/r12); раунд EGAP на lazy-атласе 33k не завершается — применять до lazy; файлы aida: ~/spore_v5/w19/ (rf_*.out, rf_*_drop.out, results собраны в v7/reports/bdp/mq/results.md); task_log писать из корня проекта; идущих расчётов нет.
Решения цепочки: нет новых (эталон случайных стартов = лучшее известное по зёрнам: ocp_arcs 0/33 допустимых).


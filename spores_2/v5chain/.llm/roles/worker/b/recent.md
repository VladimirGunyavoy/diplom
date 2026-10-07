# Последние звенья линии B

# Session Handoff — линия B
Звено: hub-v5chain-worker-b5 (acc3, sonnet, 05:02–~09:05), причина смены: ctx ~26–28% (софт 30%)
Сделано (всё в v7, коммиты b5; отчёт `v7/reports/growN/results.md`): п.31 перенос r18 в grow_cells2d (регрессия 681 кл./T 1.006 = до); growN.solve: SBCAUS (причинный граф, SBLAY=0), STGPU/SOLVEGPU (GPU aida, torch только в `~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python`), NOLATCH, SMEAN, SONE (один стенсил на группу, п.44в), DI4REF (эталон ±RHO); finish_gen: FINGRID=1/2 (shoot_grid/shoot_pol, п.42б). di4 L1600 (SBCAUS SBLAY=1 SONE NOLATCH FINGRID=2 VF1, эталон ±.35): 57/60, T/эт мед. 1.036, solve 162 с. manip c3000 (верная динамика): reach .125 (1/8), GPU solve 225 с (CPU 2248 с) — `growN_4d.md`.
Стоп на / следующий шаг: п.44 (а) метрика посева (прирост покрытия на клетку, доля уже покрытого объёма новой клетки, печатать по ходу build; стоп по приросту, не по COVTOL) и (б) посев в дырах слоя без наложений (CUT/TRIM из 2D или меньший RMAX) в `growN.py` цикл посева ~стр. 252–272; цель — покрытие слоя ≥ .9 при ≤ ~2 стенсилов на точку. Затем manip c3000+: слои без MAXC/COVTOL с GPU-конвейером (STGPU SOLVEGPU SBCAUS SONE; manip 4D, 59M рёбер помещается в GPU). п.42(а) мелкие клетки у цели — не делал (FINGRID закрыл нужду).
Грабли: ssh/aida — только через `launch_bg.py … -- ssh -p 2222 random@127.0.0.1 'bash ~/spore_v5/wb5/<скрипт>.sh'` (tqdm-фильтр grep буферит вывод до конца); эталон r22 на aida нужен в `~/spore_v5/wb4/spores_2/v5chain/reports/research/r22/` (символьная ссылка wb5/s*/v5chain → wb4); комментарий `#` внутри однострочного блока съедает остаток строки — не вставлять код после `#`; НЕ `pkill -f` на общей aida (убил бы чужие); git без identity — GIT_AUTHOR_NAME/EMAIL=gun.vladimir26@gmail.com, push не проходит; Lfull.pkl (25650 кл., 9.46M узлов) — aida `~/spore_v5/wb5/p38/`, полные слои не нужны (рабочая точка MAXC ~1600 + финиш); GPU 16 ГБ — граф ≤ ~70M рёбер (int32/float32).
Решения цепочки: SBCAUS=1 SBLAY=0 по умолчанию; SBLAY=1 только при покрытии слоя; вёдра в 4D не делать; SMEAN/SONE требуют SOLVEGPU.
Коммиты: v7 SONE, SMEAN, FINGRID, GPU; v5chain STATUS/PLAN
Токены: T_START 5ч 69% (старое окно) / ctx 5% $0.11 (05:02) / T_BEFORE_END 5ч ~50% / ctx ~27% $8.7
NEXT_LINK: hub-v5chain-worker-b6   NEXT_MODEL: sonnet

# Session Handoff — линия B
Звено: hub-v5chain-worker-b4 (acc2, sonnet, 01:21–~05:10), причина смены: ctx ~28% (софт 30%)
Сделано: п.27 SELFOV; п.28 перенос NF2/MADAPT/BFINE/SELFOV в v7 (цели воспроизведены); NF2 на u .15 и ДИ; п.30 QFAST (q_ms LOOK20 ×9.8, T те же); п.33 solve 4D — негатив (GS ×1.1, SOLVEB вёдра каскадируют, выкл.); п.35 знак Кориолиса (growN+v6) и эталон manip пересчитан. Детали — STATUS «### Линия B», v7/reports/growN/neighbors.md, nf_curved_ends.md, journals/worker/hub-v5chain-worker-b4.log.
Стоп на / следующий шаг: (1) aida: идёт manip c3000 на ИСПРАВЛЕННОЙ динамике (`~/spore_v5/wb4/growN_manip_c3000_cor.log`, DUMP `/home/random/spore_v5/wb4/manip_c3000_cor.pkl`, запуск `~/spore_v5/wb4/run370.sh`; ref — новые manip_bruteforce_{0,1,3,5}.json, скопированы и в wb1/spores_2/v5chain). Ждать JSON (reach, T/эталон по 4 стартам; build+стенсилы ~30 мин, solve ~20 мин) → STATUS, `knowledge/research/growN_4d.md`, строка research-21. Прежние manip-числа недействительны. (2) п.31 (перенос правок прототипа research-20 в grow_cells2d поверх QFAST) — не начат, см. PLAN. (3) Если manip c3000 с новой динамикой даёт reach мало — вывод по охвату 4D заново (старые выводы про 25% относились к неверной системе). (4) п.23 профиль solve 4D: Якоби 268 ит — искать сокращение числа проходов (STOL как в 2D, шаг стенсила ≥ расстояния до следующей строки) или стенсил по 2–4 вершинам, не вёдра.
Грабли: ssh с фоном — только через скрипт (`scp run.sh; ssh 'setsid nohup bash run.sh >/dev/null 2>&1 </dev/null & echo ok'` с timeout), иначе висит; `GS` в growN = число затравок от цели (по умолчанию 300 для manip), НЕ мой удалённый GS-режим; pickle DUMP из __main__ грузить, подставив классы в __main__ (solve_gs_test.py); aida ОЗУ делится, c3000 стенсилы ~15–20 ГБ; git без identity — экспортировать GIT_AUTHOR_NAME/EMAIL (gun.vladimir26@gmail.com), push не проходит (нет логина, по правилам не пушим); окружение 312 (u .3 база) не сохранено — базу сравнения ставить своим прогоном; v7 коммиты: 50e4132 (п.30), de86963 (п.33), 18d25a4 (п.35), e98d5cd (п.28), c22aebd (п.27).
Решения цепочки: c9000 PESS=1 остановлен (Кориолис), ACT/GS-режимы удалены; SOLVEB оставлен выкл.
Коммиты: v7 50e4132, de86963, 18d25a4; v6 00b321c; v5chain 04c3490
Токены: T_START 5ч 11% / ctx 5% $0.10 (01:21) / T_BEFORE_END 5ч 27% / ctx 27% $8.63
NEXT_LINK: hub-v5chain-worker-b5   NEXT_MODEL: sonnet

## hub-v5chain-worker-b3 (acc1, 12:18–01:30, ctx 28%)
manip 4D c3000g/c6000g/PESS; growN SPAR+чанки+23а×2.2+23б; п.27(а) замер самоналожения. Идёт c9000 PESS=1 на aida. Детали — STATUS «Линия B».

## hub-v5chain-worker-b1 → b2 (линия B, 04:02–07:45) и hub-v5chain-worker-b2 (07:45–~12:05, умер внезапно, без хендоффа) — линия B
Сделано b1: п.18а/18б/20/21 в v7, п.19 эталон u .3 ✓ (u .15 позже сделал research-17). b2: v7/src/cells7/growN.py (n-мерный grow3: dd/pend/manip), (а) dd 54/60 T/эт 1.0059 ✓, (б) pend n=2 1.055 ✓, п.22г финиш VF встроен, п.23 частично (HexIdx 105→13 с), goal_seeds для manip.
Стоп: (в) manip 4D — MAXC=3000 с затравками от цели: reach .25 (2/8), T/эт 1.24 (VF 1.0: 1.18); полное покрытие 4D не сходится (очередь >10^5 клеток). Грабли: aida в ssh — ставить фон без stdout-привязки; tail лога tqdm — только через tr "\r" "\n" и tail -c; после конца расчёта висят процессы пула growN — убивать по PID.

## hub-v5chain-worker-b1 (линия B, 2026-10-06 04:02–07:45): перенос в v7 OWN/CUTR/LOOK/NORMFRONT/SEEDEPS (сверено с r16), эталон маятника pend_ref_best.py (u .3 ✓, u .15 считается на aida); дальше PLAN 22 growN. Детали — HANDOFF_b.md.

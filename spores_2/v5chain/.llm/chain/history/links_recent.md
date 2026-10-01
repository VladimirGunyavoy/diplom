# links_recent — последние 5 звеньев (новые сверху)

## hub-worker-10 (acc3, sonnet, 16:44–21:40), смена: human (NEEDS-HUMAN п.3)
v7: locate предфильтр ×2, симметричная проверка ядер, nD-клетка celln.py, покрытие 9 систем (v7_summary.md), п.5 эллипс+пул (ellipse.py), картинки dd/m2g. V по клеткам (граф/решётка/гало) не работает — v7_cells.md. Грабли: v6 импортируется по именам файлов; nD-метрики O(N²), 8D≈20 мин; sleep>120 блокируется; git -c user.email обязателен; не пушить. Код v7 на aida ~/spore_v5/w10/v7.

## hub-worker-9 (acc3, sonnet, 09:20–16:50), смена по ctx 21%
`corridor_query(fs,neigh,wlim)`, refine |w|≤WM; v6_summary (WM3/6, neigh8, dd+диски 40/40, маятник, двойной маятник g=1/1.5/2); v7 каркас `spores_2/v7/src/cells7` (systems, cell, cover) + tests/run_cover.py. Грабли: ssh -f + setsid nohup; sleep>120 блокируется; casadi нет; pkill -f убивает шелл; git add только своих путей. Код v7 на aida `~/spore_v5/w9/v7`.

## hub-worker-8 (acc2, sonnet, 02:40–09:30), смена по ctx 22%
ночные эксперименты (а)–(д): статистика 3зв/2зв/dd/маятник, 8D 100k 8/8, sweep, адаптив vs сетка, `v6/reports/v6_summary.md`; `back_heuristic` WH=1.5 при препятствиях. Грабли: aida `~/spore_v5/w7/v6` (scp -r src tests); `OMP_NUM_THREADS=1 nohup`, фоновый until; env тестов G,DOWN,UP,DT,WH,KN,WM,NQ,NAME,OBST.


## hub-worker-7 (acc3, sonnet, 16:30–02:45), смена по ctx 22%
corridor_nd maxiter 80, corridor_batch (Pool), tries=, hfun= (A*); manip3dyn.py (3 звена 6D+g+диски). 6D без g 4/4; 3зв.+g 4/4; 2зв.+g 8/8; +2 диска 3/4; A* помогает без препятствий, с дисками хуже (1/4 vs 2/4). Грабли: SLSQP хаотичен (T ±15%); 6D+g dt_max≤.02; код на aida ~/spore_v5/w7; pkill -f убивает шелл; лог aida в файл.

## hub-worker-6 (acc2, sonnet, 08:40–16:35), смена по ctx 21%
коридор nD `corridor_nd.py`, `replay_value_fast` ×11.7, манипулятор 4D динамика `manip2dyn.py` (= эталон research, с дисками 8/8), скалярный flow4 (7–72 с/запрос), дд с дисками 10/10. Грабли: pkill -f убивает свой шелл; вывод фона — в файл; miss(X) векторный; ns зазора ≥16 точек/сегмент; NB≈2400 для 4D; временные скрипты удалять.


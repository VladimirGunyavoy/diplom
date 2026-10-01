# links_recent — последние 5 звеньев (новые сверху)

## hub-worker-11 (acc3, sonnet, 20:10–00:40), смена: ctx 28%
v7: градиент V (value_jet.py) на DI — не сошёлся (v7_cells.md); hold.py/holdn.py спектр+LQR (DI/маятник/манип. 2 зв.; v7_hold.md); данные для research-7 (exits_summary.md); **faces.py — общие грани по research-7 (v7_faces.md): DI 1.025 при 82 линиях, маятник — достижимость падает на мелких сетках (открыто)**. Грабли: pkill -f убивает шелл; тяжёлое — systemd-run MemoryMax=3G; sleep>120 блокируется (until-цикл); git -c user.email/name обязателен; не пушить.

## hub-worker-10 (acc3, sonnet, 16:44–21:40), смена: human (NEEDS-HUMAN п.3)
v7: locate предфильтр ×2, симметричная проверка ядер, nD-клетка celln.py, покрытие 9 систем (v7_summary.md), п.5 эллипс+пул (ellipse.py), картинки dd/m2g. V по клеткам (граф/решётка/гало) не работает — v7_cells.md. Грабли: v6 импортируется по именам файлов; nD-метрики O(N²), 8D≈20 мин; sleep>120 блокируется; git -c user.email обязателен; не пушить. Код v7 на aida ~/spore_v5/w10/v7.

## hub-worker-9 (acc3, sonnet, 09:20–16:50), смена по ctx 21%
`corridor_query(fs,neigh,wlim)`, refine |w|≤WM; v6_summary (WM3/6, neigh8, dd+диски 40/40, маятник, двойной маятник g=1/1.5/2); v7 каркас `spores_2/v7/src/cells7` (systems, cell, cover) + tests/run_cover.py. Грабли: ssh -f + setsid nohup; sleep>120 блокируется; casadi нет; pkill -f убивает шелл; git add только своих путей. Код v7 на aida `~/spore_v5/w9/v7`.

## hub-worker-8 (acc2, sonnet, 02:40–09:30), смена по ctx 22%
ночные эксперименты (а)–(д): статистика 3зв/2зв/dd/маятник, 8D 100k 8/8, sweep, адаптив vs сетка, `v6/reports/v6_summary.md`; `back_heuristic` WH=1.5 при препятствиях. Грабли: aida `~/spore_v5/w7/v6` (scp -r src tests); `OMP_NUM_THREADS=1 nohup`, фоновый until; env тестов G,DOWN,UP,DT,WH,KN,WM,NQ,NAME,OBST.


## hub-worker-7 (acc3, sonnet, 16:30–02:45), смена по ctx 22%
corridor_nd maxiter 80, corridor_batch (Pool), tries=, hfun= (A*); manip3dyn.py (3 звена 6D+g+диски). 6D без g 4/4; 3зв.+g 4/4; 2зв.+g 8/8; +2 диска 3/4; A* помогает без препятствий, с дисками хуже (1/4 vs 2/4). Грабли: SLSQP хаотичен (T ±15%); 6D+g dt_max≤.02; код на aida ~/spore_v5/w7; pkill -f убивает шелл; лог aida в файл.

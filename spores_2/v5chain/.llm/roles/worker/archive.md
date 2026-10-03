## hub-worker-9 (acc3, sonnet, 09:20–16:50), смена по ctx 21%
`corridor_query(fs,neigh,wlim)`, refine |w|≤WM; v6_summary (WM3/6, neigh8, dd+диски 40/40, маятник, двойной маятник g=1/1.5/2); v7 каркас `spores_2/v7/src/cells7` (systems, cell, cover) + tests/run_cover.py. Грабли: ssh -f + setsid nohup; sleep>120 блокируется; casadi нет; pkill -f убивает шелл; git add только своих путей. Код v7 на aida `~/spore_v5/w9/v7`.

## hub-worker-8 (acc2, sonnet, 02:40–09:30), смена по ctx 22%
ночные эксперименты (а)–(д): статистика 3зв/2зв/dd/маятник, 8D 100k 8/8, sweep, адаптив vs сетка, `v6/reports/v6_summary.md`; `back_heuristic` WH=1.5 при препятствиях. Грабли: aida `~/spore_v5/w7/v6` (scp -r src tests); `OMP_NUM_THREADS=1 nohup`, фоновый until; env тестов G,DOWN,UP,DT,WH,KN,WM,NQ,NAME,OBST.


# links_archive
- сборщик v5chain (acc2, Opus, 2026-09-29): собрал v5chain = код v4_ort + `.llm`; цель записана в TASK/PLAN. Грабли: скрипты system/infra берут проект из cwd; `src/atlas/` — отдельный пакет, Ursina не импортировать; сбой классификатора «no verdict» — временный, повторить.

- hub-worker-4 (acc2, Sonnet, 2026-09-30): v6 дифдрайв `dd_atlas.py` (ромб/RECT, агент, коридор SLSQP, диски), манипулятор `manip2.py` (V==Дейкстра), H1 DI `adaptive_di.py` (N≈117 V/T*=1.02 vs решётка N=2401 1.30). Грабли: cron/sid/журнал только в v5chain; pgrep -f в until находит сам себя; sleep>120 блокируется; коммит -c user.name/-c user.email, git add только своих путей; rm -r блокируется.

## hub-worker-5 (acc2, sonnet, 03:20–08:45), смена по ctx 20%
solve_V починен; H1 подтверждена: adaptive_tree.py/adaptive_nd.py (DI 1.03 N≈60, маятник 0.97 ~400, dd 3D 10/10 1.02–1.27, кинематика n=2–4 1.0–1.2). Факты — knowledge/v6_findings.md. Грабли: cron/sid/журнал только в v5chain; pkill -f "python3 -" убивает шелл; sleep>120 блокируется; git -c user.name/-c user.email, add только своих путей; gamma-тесты нужен NB/rho.

- hub-research-1 (acc3, Opus, 2026-09-29/30): H1 (адаптив ≈ сетка при ~20× меньше спор: DI 117 vs 2401, маятник ~400 vs 7776), H2 (выравнивание цепочек, сдвиг 1% → +28%), эталоны dd (min(TGT,TGTGT), ref_window2), стыковка nD (`docking_nd.md`). Грабли: RESEARCH_HANDOFF в .llm/state игнорируется git — `git add -f`; SLSQP-эталоны медленные — в фон.
- hub-worker-3 (acc3, Sonnet, 2026-09-29): v6 — DI (клетка, атлас, агент, V, коридор T*=1.0000, AtlasView клавиши 9/0), маятник u=.5/.3, клетка 3D дифдрайва dd3.py. Грабли: Ursina на хабе без окна; pytest нет; git add только своих путей; ρ по длинам рёбер; поле агента DI ±4; клетка цели обязательна.
- hub-worker-2 (acc1, Sonnet, 2026-09-29): схема B 5D (`five_d.py`, rollout5_best 40/40, snap, чётность подрешётки), A vs B, хвост DI у vmax, создана роль research (hub-research-1, Opus, 30 мин). Грабли: aida `ssh -n -p 2222 random@127.0.0.1`, код `~/spore_v5` (scp, не tar|ssh), запуск `(nohup … &)`; НЕ `pkill -f`; sleep >120 с уходит в фон; pytest нет; коммит с -c user.name/-c user.email.
- hub-worker-1 (acc1, Sonnet, 2026-09-29): атлас DI SOLVED; дифдрайв схема A: A1–A4, A2b, A5-lite, сводка `reports/diffdrive_atlas.md`. Грабли: pytest нет — `python3 tests/test_*.py`; коммит с `-c user.name/-c user.email`; arc_sweeps n=6 ≈17 с; rollout на поле без дуг застревает; после 07:40 простаивал — при отсутствии ответа брать следующий пункт PLAN сразу.

## hub-worker-6 (acc2, sonnet, 08:40–16:35), смена по ctx 21%
коридор nD `corridor_nd.py`, `replay_value_fast` ×11.7, манипулятор 4D динамика `manip2dyn.py` (= эталон research, с дисками 8/8), скалярный flow4 (7–72 с/запрос), дд с дисками 10/10. Грабли: pkill -f убивает свой шелл; вывод фона — в файл; miss(X) векторный; ns зазора ≥16 точек/сегмент; NB≈2400 для 4D; временные скрипты удалять.

## hub-worker-7 (acc3, sonnet, 16:30–02:45), смена по ctx 22%
corridor_nd maxiter 80, corridor_batch (Pool), tries=, hfun= (A*); manip3dyn.py (3 звена 6D+g+диски). 6D без g 4/4; 3зв.+g 4/4; 2зв.+g 8/8; +2 диска 3/4; A* помогает без препятствий, с дисками хуже (1/4 vs 2/4). Грабли: SLSQP хаотичен (T ±15%); 6D+g dt_max≤.02; код на aida ~/spore_v5/w7; pkill -f убивает шелл; лог aida в файл.

## hub-worker-14 (acc3, Sonnet, 00:20–19:30, ctx27%)
Звено: hub-worker-14 (acc3, Sonnet, 00:20–19:30), причина смены: ctx ~27% (софт 30%, рабочая смена ~19 ч)
Сделано: очередь PLAN п.1–5 + задачи research-9 (всё в `knowledge/research/butterfly_spores.md`, `butterfly_dd.md`, `v7/reports/spore_v_di.md`): гибрид бабочки маятника + финиш стрельбой (VF3 1.028; u·.95 на V с заполненными дырами 1.070, 99%); EST=3 на spore_v с шумом (σ.01 1.02); бабочки ДИ на общих стартах (1.064 vs 1.044); 4D бабочки 50k спор (V med 1.092, агент med 1.101, финиш по кривой переключения VF2 → max 1.49); дифдрайв бабочки в v7 (1.033) + 2 диска (100%, 0 столкновений, 1.077–1.085 vs свободный эталон); g=2 тёплый старт OCP — отрицательно. PLAN.md ротирован 53→8 КБ.

## hub-worker-13 (2026-10-02, acc2 Sonnet, ctx 28%)
Звено: hub-worker-13 (acc2, Sonnet, 10:11–02:45), причина смены: ctx 28% (софт 30%, идл)
Сделано: spore_v исправлен и измерен (отчёт `v7/reports/spore_v_di.md`, всё в PLAN): `Cell.locate` (допуск гало, экстраполяция Эрмита, предфильтр по t → 99.9% узлов находят свою клетку), `dt_edge`=шаг агента, густота узлов: DI T/T* 1.044 (5→3 перекл.), маятник hs=.008 T/Ta 1.019; LQR-зона hc=.1; спектр u в агенте ≤1%; точное переключение (research-8/9) на spore_v: P3:.5+EST=1+EPS=.02 sw 18→3, u·.95: 1.027; бабочки (research-8): ДИ ×100 (`cells7/butterfly_di.py`), маятник 100%/1.036/4 перекл. при 35k узлов (`cells7/butterfly_pend.py`, окно .3); против строгого эталона pend_ref_T: spore_v 1.075, бабочки 1.077.
Грабли: `pkill -f` убивает шелл (дважды наступил) — kill по pid; на хабе всего 7 ГБ RAM — тяжёлое с `systemd-run --user --scope -p MemoryMax=4G`, в фоне `setsid nohup … & disown`; V.build у маятника hs=.008 ~11 мин, SporeV(...) init ~5–10 мин; кэши /tmp/claude-1000/{pend_V6.npy (hs.008,dt_edge.06), bp_V4.npy (бабочки окно .3), pend_cells_filled.pkl, di_cells_filled.pkl}; скрипты /tmp/claude-1000/{q*,s9,bp*,es_run*,es_head*}.py (в /tmp — могут пропасть); git: нет user.name → `git -c user.name=hub-worker-N -c user.email=a@b commit`; STATUS велит не пушить — не пушил; бабочки: агент без wrap φ накручивает обороты.
Решения цепочки: рекомендуемые настройки агента spore_v — history/decisions.md.

# links_recent — последние 5 звеньев (новые сверху)

## hub-worker-12 (acc1, Sonnet, 00:08–10:15), причина смены: ctx 28% (софт 30%)
Сделано: маятник на faces: привязка проб к линиям + `bigfin` (по диагнозу research-7) → 100% дошли, T/Ta 1.001 (v6 PendAtlas); затем по рекомендации research-7 написан `v7/src/cells7/spore_v.py` (V по клеткам cover_layer+гало, V*=min_k[Δt+V*(φ_k)], адаптивные узлы, `fill_gaps`, периодичность): DI 100%, T/T* 1.15 (5 перекл.), маятник 98%, T/Ta med 1.32 (18). Отчёт `v7/reports/spore_v_di.md`, скрипты+кэши `v7/reports/spore_v/` (в скриптах пути /tmp/claude-1000 — подправить).
Стоп на / следующий шаг: (1) **идёт** `/tmp/claude-1000/s8.py` → `s8.log` (DI с `blend=True` — взвеш. среднее вместо min; гипотеза: min по клеткам смещает V вниз, V*<T* у 97% узлов, V/T* med .87); если жив — забрать итог (V*/T* med, T/T*, dt .06/.02), иначе перезапустить (~40+ мин, bincount); (2) LQR-клетка у цели маятника (`hold.py`, sat(−Kx), xᵀPx≤.5, research-7: скачки у цели 15→0) в агент spore_v; (3) спектр u (11 значений) / L1-штраф (v7_spectrum_cost.md): в Беллмане и агенте непрерывный набор; (4) ответ hub-research-7 на сверку spore_v vs v7_spore_di.py (отправил список расхождений; его гипотеза (а) шаг агента=шагу Беллмана НЕ подтвердилась: dt .06→1.13, .02→1.49, .0075→2.68); (5) адаптивность граней (faces) — отложена рекомендацией research-7.
Грабли: `pkill -f` убивает шелл — убивать по pid; `np.add.at` в Якоби катастрофически медленный (использовать bincount); cover_layer m=40 ≈ 1000 с на слой — не пересчитывать, брать pkl; 5% точек вне всех клеток (дыры) → V=BIG, агент встаёт: всегда `fill_gaps`; агент маятника накручивает обороты — `S.wrap` в `_pairs` (иначе BIG); гистерезис eps=.003 не помог (DI, маятник); git index.lock бывает от чужих агентов — повторить; тяжёлое только через `systemd-run --user --scope -p MemoryMax`, в фоне `setsid nohup … & disown`; фоновые прогоны без setsid умирают молча.
Решения цепочки: гипотеза «физический предел» маятника отозвана (слово пользователя); faces→spore_v по рекомендации research-7 (v7_spore_cells.md).
Коммиты: v7: cb67026, ce614a2, 2eb84ca…; цепочка: PLAN/STATUS обновлены (не пушил: STATUS «не пушить»).
Токены: T_START 5ч 17% / ctx 4% / $0.10 (00:08) / T_BEFORE_END 5ч 25% / ctx 27% / $10.1 (10:15)
NEXT_LINK: hub-worker-13   NEXT_MODEL: sonnet

## hub-worker-11 (acc3, sonnet, 20:10–00:40), смена: ctx 28%
v7: градиент V (value_jet.py) на DI — не сошёлся (v7_cells.md); hold.py/holdn.py спектр+LQR (DI/маятник/манип. 2 зв.; v7_hold.md); данные для research-7 (exits_summary.md); **faces.py — общие грани по research-7 (v7_faces.md): DI 1.025 при 82 линиях, маятник — достижимость падает на мелких сетках (открыто)**. Грабли: pkill -f убивает шелл; тяжёлое — systemd-run MemoryMax=3G; sleep>120 блокируется (until-цикл); git -c user.email/name обязателен; не пушить.

## hub-worker-10 (acc3, sonnet, 16:44–21:40), смена: human (NEEDS-HUMAN п.3)
v7: locate предфильтр ×2, симметричная проверка ядер, nD-клетка celln.py, покрытие 9 систем (v7_summary.md), п.5 эллипс+пул (ellipse.py), картинки dd/m2g. V по клеткам (граф/решётка/гало) не работает — v7_cells.md. Грабли: v6 импортируется по именам файлов; nD-метрики O(N²), 8D≈20 мин; sleep>120 блокируется; git -c user.email обязателен; не пушить. Код v7 на aida ~/spore_v5/w10/v7.

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

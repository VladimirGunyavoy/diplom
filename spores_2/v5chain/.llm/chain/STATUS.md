# STATUS — снимок цепочки (≤ 80 строк, заменять, не дописывать)
Коридор ctx: по умолчанию (`~/claude-work/system/llm/ARCHITECTURE_CURRENT.md` §Коридоры); активный оверрайд — писать этой строкой.
Модель воркеров — **Sonnet** (слово пользователя 2026-09-29); Opus — изредка, следить за недельным расходом (пользователю нужны токены на пятницу). Работа только на хабе; не пушить.

## Сейчас
- Атлас двойного интегратора — SOLVED (`reports/atlas_double_integrator.md`, `reports/improvements.json`, `src/atlas/`, `src/atlas/improve.py`).
- Дифдрайв, схема A (v, ω — управления; решение агента 2026-09-29, `history/decisions.md`): `src/atlas_dd/` (modes, lattice, reference, plan), тесты `tests/test_atlas_dd_*.py`, сводка `reports/diffdrive_atlas.md`, факты `knowledge/diffdrive_atlas_design.md`.
  Финиш rollout_multi: xy 0.011, θ<0.005; дуги (интерполяционные) по h НЕ сошлись.
- A2c `arc_edges` (дуги в узле), схема B 5D `src/atlas_dd/five_d.py` (+`rollout5`), диагноз несходимости по h = интерполяция T вне узлов; всё в `reports/diffdrive_atlas.md`. Пользователь: не ждать, работать непрерывно.
- Схема B: snap к узлу в rollout5 (5/6 стартов в цель), dt-сходимость, поля n=16/24 (считалось на aida, `~/spore_v5`, по слову пользователя) — всё в `reports/diffdrive_atlas.md`.
- Схема B доведена: rollout5_best 40/40 (v,ω≠0), 64/64 граница, привязка к достижимой подрешётке (чётность), A vs B по времени B/A≈2.4; хвост DI: запросы у vmax (`knowledge/atlas_di_tails.md`). Новая роль research (`hub-research-1` поднят, acc3, Opus) — `RESEARCH_ROUTINE.md`.
- Коммиты локальные, пуша нет.

- **v6 (сейчас):** `spores_2/v6/src/atlas6/`: DI/маятник (cell, atlas, agent, value, corridor, gcell, pend, gcorridor), дифдрайв `dd3.py`+**`dd_atlas.py`** (ромб 4 слоя/RECT 8, τ=h, V от верхней оценки, агент, коридор SLSQP+откат, препятствия-диски; эталон min(TGT,TGTGT) mean 1.05), манипулятор 2 звена кинематика **`manip2.py`** (тор, препятствия, V==Дейкстра), **`adaptive_di.py`** (H1: адаптив N≈117 V/T*=1.02 vs решётка N=2401 1.30, один запрос). Тесты `tests/test_atlas6_*.py`, проверки `tests/check_*.py`. Факты — `knowledge/v6_findings.md`.

- **H1 (адаптивный атлас vs сетка) — подтверждена (hub-worker-5):** `v6/src/atlas6/adaptive_nd.py` (дерево цепочек с отсечением по ядрам, стык проигрышем, любая размерность, препятствия `blocked`) и `adaptive_tree.py` (2D: build_tree/cross_value). DI N≈60 V/T* 1.03; маятник u=.3 ~400 спор 0.97 (сетка 7776: 1.04); дифдрайв 3D 600+600 спор 10/10 V/ref_window2 1.02–1.27; кинематика n-звенника n=2–4 (T* точный) 1.0–1.2, покрытие падает с n. Факты — `knowledge/v6_findings.md`.

- **research (hub-research-2, 2026-09-30):** коридор на адаптивном дереве = эталон: дд 3D NB1200+NF10 → 30/30 T/ref 1.004; маятник 8/8; манипулятор 4D NB2400/NF400 8/8 = перебору (`knowledge/research/multiquery_corridor.md`, `manip_dyn_corridor.md`); правила в `src/atlas6/corridor_nd.py` (worker-6).
- **research (hub-research-3, 2026-09-30): почему запросы медленные** — ~98% в SLSQP `refine` (numpy-rk4 точкой; math ×13–22, maxiter 80, зазор последовательно): манипулятор на aida 49–60 с → 1.24 с при тех же T; хаб ≈10× медленнее aida (VM). `knowledge/research/query_speed.md`, рекомендации — PLAN.

- **Коридор nD + манипулятор 4D (hub-worker-6):** `v6/src/atlas6/corridor_nd.py` (топологии дерева → SLSQP: кандидаты с промахом, top-K попаданий+K промахов, локальный поиск соседей, зазор до препятствий `clear`, батч-`candidates`), `replay_value_fast` (×11.7), `manip2dyn.py` (динамика 2 звена, `clearance`, скалярный flow4). Результаты: дд 30/30 T/ref 1.000; дд с дисками 10/10 без столкновений; манипулятор 4D NB2400 NF400 = эталон research (q0,q1,q3,q5), с 2 дисками 8/8 без столкновений, 7–72 с/запрос; маятник u=.3 7/8 mean .954 (q2 без допустимой топологии). Ursina на хабе не запускается (нет графики), импорты чистые. Всё — `knowledge/v6_findings.md`.

- **hub-worker-7 (2026-09-30/10-01):** `corridor_nd`: maxiter 80 (T те же), `corridor_batch` (Pool; дд 30/30 за 8 с хаб / 0.4 с aida), параметр `tries`. `manip3dyn.py`: 3 звена 6D (+гравитация g, скалярный flow ×4, `clearance` для дисков). Коридор: 6D без g 4/4 (NB12000 NF3000 DT.02); 3 зв.+g вниз 4/4; 2 зв.+g вниз 8/8, вверх 8/8 (NB3000 NF2000); 3 зв.+g +2 диска 3/4 (NB24000 NF6000, 132–198 с). Нерешённое — нехватка покрытия (NB/NF лечит), кроме q4 с дисками (вероятно неизбежное столкновение). Код на aida: `~/spore_v5/w7`. Факты — `knowledge/v6_findings.md`.

- **hub-worker-8 (2026-10-01):** «Эксперименты на ночь» (а)–(д) выполнены (aida, corridor_batch): `v6/reports/v6_stats.md` + `v6_summary.md` + `stats_*.json`. 3 зв.+g 6D NB12000 NF3000: вниз 32/32 T/ref med 1.018 max 1.17, вверх 32/32 med 1.036 max 1.36 (WM=6: mean 1.085→1.057); +2 диска 26/32; 2 зв.+g 40/40; dd+диски 38/40; маятник 40/40; 8D 4 звена NB12k 3/8, 40k 7/8, 100k 8/8; адаптив vs сетка 4D (N=2400 1.14× против сетки 147k 2.26×). A* с дисками: WH=1–2 (дефолт 1.5 при `has_obst`). q4+диски — неизбежное столкновение. Потери вверх: ветвь 2π (⅔) + рамка WM (⅓), K/tries не лечат.

- **hub-worker-9 (2026-10-01):** `corridor_query(fs,neigh,wlim)`; |w|≤WM физический в refine (0/64 нарушений); 3 зв.+g вверх WM6+neigh8 mean 1.041 (WM3 1.085); dd+диски 40/40 при NB4800 NF600; двойной маятник g=1 32/32 (вниз→вверх T 5.28), g=2 только при WM6; 8D без выигрыша; **v7** (`spores_2/v7/src/cells7`: клетки с локальной моделью, ядро+гало, покрытие) — каркас, DI считается на aida. Всё — `v6/reports/v6_summary.md`.

- **research (hub-research-6, 2026-10-01):** спектр управлений (`knowledge/research/control_spectrum.md`: 3 слоя/канал, у цели спектр+LQR держит, вырождение слоя → клетка по образу U); ветвь 2π вверх — только 2/5 худших (`branch2pi.md`); слепая DI-эвристика h/V 0.41/0.59 (`blind_di_heuristic.md`); модель поля по q отклонена.

- **hub-worker-12 (2026-10-02, v7 п.3 дальше):** `v7/src/cells7/faces.py` (грани; маятник сверен с v6 PendAtlas T/Ta 1.001, `bigfin=True`) → заменено рекомендацией research-7 на `v7/src/cells7/spore_v.py` (V по клеткам спор с гало, адаптивные узлы, `fill_gaps`, периодичность): DI 100% дошли T/T* 1.15 (5 перекл.), маятник 98% T/Ta med 1.32 (18 перекл.); отчёт `v7/reports/spore_v_di.md`, скрипты/кэши `v7/reports/spore_v/`. Открыто: смещение V (оптимистичен ~13%), режим `blend` (не завершён), LQR у цели маятника, спектр/L1 (PLAN).

## Открытые проблемы
- H1 без общих цепочек и без коридора (solve_V починен hub-worker-5).
- A/B закрыт: делаем обе схемы.
- Цена с дугами не сошлась по h (`knowledge/diffdrive_atlas_design.md`).

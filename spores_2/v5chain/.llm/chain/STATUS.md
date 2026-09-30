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

## Открытые проблемы
- H1 без общих цепочек и без коридора (solve_V починен hub-worker-5).
- A/B закрыт: делаем обе схемы.
- Цена с дугами не сошлась по h (`knowledge/diffdrive_atlas_design.md`).

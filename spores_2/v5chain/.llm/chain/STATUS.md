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

- **v6 (сейчас):** `spores_2/v6`: `src/atlas6/` (cell, atlas, agent, value, corridor, gcell, pend, gcorridor, dd3), тесты `tests/test_atlas6_*.py` (python3, pytest нет), `src/spores/atlas_view.py` + клавиши 9/0 в `main.py`. DI: коридор время/T*=1.0000; маятник: агент≈V, коридор на 2–6% короче мелкой V; дифдрайв: клетка 3D (L_q-инвариантность 4e-16). Факты — `knowledge/v6_findings.md`.

## Открытые проблемы
- A/B закрыт: делаем обе схемы.
- Цена с дугами не сошлась по h (`knowledge/diffdrive_atlas_design.md`).

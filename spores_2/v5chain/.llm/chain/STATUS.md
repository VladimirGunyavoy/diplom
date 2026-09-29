# STATUS — снимок цепочки (≤ 80 строк, заменять, не дописывать)
Коридор ctx: по умолчанию (`~/claude-work/system/llm/ARCHITECTURE_CURRENT.md` §Коридоры); активный оверрайд — писать этой строкой.
Модель воркеров — **Sonnet** (слово пользователя 2026-09-29); Opus — изредка, следить за недельным расходом (пользователю нужны токены на пятницу). Работа только на хабе; не пушить.

## Сейчас
- Атлас двойного интегратора — SOLVED (`reports/atlas_double_integrator.md`, `reports/improvements.json`, `src/atlas/`, `src/atlas/improve.py`).
- Дифдрайв, схема A (v, ω — управления; решение агента 2026-09-29, `history/decisions.md`): `src/atlas_dd/` (modes, lattice, reference, plan), тесты `tests/test_atlas_dd_*.py`, сводка `reports/diffdrive_atlas.md`, факты `knowledge/diffdrive_atlas_design.md`.
  Финиш rollout_multi: xy 0.011, θ<0.005; дуги (интерполяционные) по h НЕ сошлись.
- Коммиты локальные (последний 914f777), пуша нет.

## Открытые проблемы
- Вопрос пользователю A/B (v, ω — управления или часть состояния, 5D) — `JOURNAL.md` NEEDS-HUMAN; ответа нет.
- Цена с дугами не сошлась по h (`knowledge/diffdrive_atlas_design.md`).

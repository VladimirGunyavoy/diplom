# Changelog Archive - История изменений

> Архив. Актуальные последние сессии → [changelog_recent.md](changelog_recent.md)

---

## 2026-04-19 (сессия 8) - SharedContext + GhostSpore + TauManager

**Что сделано:**
- 🆕 `src/shared_context.py` — универсальный контейнер живых данных
- 🆕 `src/tau_manager.py` — параметр τ
- 🆕 `GhostSpore` — следует за `ctx.look_point` каждый кадр
- 🔄 `zoom_manager.py` — добавлен `real_look_point` property
- 🔄 `spore_manager.py` — `List` → `Dict[str, Spore]`, добавлен `get(name)`
- 🐛 Исправлена опечатка `positions=` → `position=` (Issue #6)

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-04-19 (сессия 7) - ScreenManager + рефакторинг биндингов

**Что сделано:**
- 🆕 `src/screen_manager.py` — `ScreenManager` + `Message` (динамический текст)
- 🔄 `src/object_manager.py` — `bind()` требует `key` и `description`; `get_help()`
- 🔄 `src/input_manager.py` — `input_frozen` блокирует все биндинги

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-04-19 (сессия 6) - Замена DiffDrive → DoubleIntegrator

**Что сделано:**
- 🔄 `src/math/diff_drive.py` → удалён
- 🆕 `src/math/double_integrator.py` — 2D double integrator, state=[x,y,vx,vy], control=[ux,uy]
- 🔄 `src/math/__init__.py` — экспортирует `DoubleIntegrator` вместо `DiffDrive`
- 🔄 `src/trajectories.py` — переписан под DoubleIntegrator
- 🔄 `main.py` — обновлён маппинг `(x, vx, y)`

**Технические детали:**
Точное аналитическое интегрирование: `x_new = x + vx*dt + 0.5*ux*dt²`

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-03-19 (сессия 5) - Trajectory visualization: Spore + ScalableLine

**Что сделано:**
- 🌱 `src/spore.py` — статичный маркер (quad + billboard), наследник MyObject
- 🎛️ `src/spore_manager.py` — управление размером всех спор (клавиши 3/4, ×1.2)
- 〰️ `src/scalable_line.py` — линия между двумя точками, реагирует на zoom
- 📐 `src/trajectories.py` — генерация DiffDrive-траекторий

**Участники:** Пользователь + Claude Sonnet 4.6

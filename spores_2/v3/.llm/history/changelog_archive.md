# Changelog Archive - История изменений

> Архив. Актуальные последние сессии → [changelog_recent.md](changelog_recent.md)

---

## 2026-04-24 (сессия 11) - Рёбра графа + BoundaryRayFamily

**Что сделано:**
- 🐛 `run.py`, `watcher.py` — фикс путей (PROJECT_ROOT указывал на `src/` вместо `player_zoom/`)
- 🐛 `color_manager.py` — фикс пути к colors.json
- 🆕 `config/colors.json` — создан со всеми цветами
- 🆕 `src/core/line_manager.py` — фабрика `ScalableLine`
- 🔄 `ghost_spore_family.py` — рёбра time/control/root→gen1 через `_GhostLineFamily`
- 🆕 `src/spores/boundary_ray_family.py` — `BoundaryRay` + `BoundaryRayFamily`
- 🔄 `shared_context.py` — менеджеры как прямые поля
- 🔄 `main.py` — `family_b` (time_sign=-1), `ray_family_plus/minus`

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-04-23 (сессия 10) - GhostSporeFamily + архитектура менеджеров

**Что сделано:**
- 🔄 `SporeManager` — добавлен `create(cls, name, **kwargs)` как прокси к ObjectManager; убрана зависимость ObjectManager → SporeManager
- 🔄 `ObjectManager` — принимает `shared_context`, auto-inject `ctx` для Spore-субклассов, добавлен `register_tickable()` для не-GameObject объектов с tick()
- 🔄 `ParamManager` — добавлены `min_val`/`max_val` с clamping; новые параметры `a_max` (кл. 3), `n_tau` (кл. 4), `n_u` (кл. 5)
- 🔄 `SharedContext` — `param_manager` забиндан как единая точка доступа к параметрам
- 🔄 `DoubleIntegrator` — рефакторинг: stateless `step(x0, v0, u, t)`, принимает SharedContext, `tick()` синхронизирует `a_max`, убрано внутреннее состояние позиции
- 🆕 `src/spores/ghost_spore_family.py` — `GhostSporeFamily`: сетка `n_tau × (2*n_u+1)` призрачных спор, рекурсивная эволюция через DI

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-04-23 (сессия 9) - Рефакторинг архитектуры: tick/register/ParamManager/структура src/

**Что сделано:**
- 🔄 `register(**kwargs)` — универсальный метод регистрации вместо отдельных `register_X()` в InputManager и UpdateManager
- 🔄 `tick()` — унифицированное имя per-frame метода у всех компонентов
- 🆕 `src/core/param_manager.py` — `ParamManager`: именованные float-параметры, exp/linear режимы
- 🗑️ `src/tau_manager.py` — удалён, tau теперь `param_manager.add('tau', 0.5)`
- 🔄 `src/core/input_manager.py` — `bind()` с mode='press' и mode='scroll', hold+scroll подавляет зум
- 🔄 `src/` реорганизована: `core/`, `spores/`, `math/`, `utils/`
- 🔄 `SceneSetup` → `SceneManager`

**Участники:** Пользователь + Claude Sonnet 4.6

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

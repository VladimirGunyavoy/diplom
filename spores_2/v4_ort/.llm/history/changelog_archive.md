# Changelog Archive - История изменений

> Архив. Актуальные последние сессии → [changelog_recent.md](changelog_recent.md)

---

## 2026-05-26 (сессия 13) - Диагностика ScalableFanSurface

**Что сделано:**
- 🔄 `src/core/scalable_surface.py` — `mode='triangle'` + явные tris → `mode='ngon'` без explicit triangles
- 🔄 `src/core/surface_manager.py` — добавлен `double_sided=True`
- 🔄 `src/spores/boundary_ray_family.py` — boundary переписан на семантически чистую версию: `positions[0,:]` + `positions[1:, n-1]` + `positions[-1,-2::-1]`
- 🆕 `tests/test_boundary.ipynb` — ноутбук для визуализации boundary (matplotlib 2D); **подтверждено: граница логически верна**

**Результат:** частичное улучшение (1 треугольник → ромб), но bug не закрыт. `double_sided` не помог.

**Следующий шаг:** вернуться к `mode='triangle'` + переприсваивать `self._tris` в каждом `apply_transform`; добавить debug-print для диагностики.

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-05-26 (сессия 12) - BranchFamily + SurfaceManager + monitors.json

**Что сделано:**
- 🆕 `config/monitors.json` — конфиг мониторов вынесен из WindowManager; добавлено поле `margin` (для левого монитора `[-0.07, -0.0]`)
- 🔄 `src/core/window_manager.py` — читает monitors.json, добавлен `get_margin()`, удалён хардкод `MONITORS`
- 🔄 `main.py` — UI-элементы используют `window_manager.get_margin()`
- 🗑️ `src/spores/ghost_spore_family.py` — удалён целиком (семья заменена веткой)
- 🔄 `src/spores/boundary_ray_family.py` — полностью переписан как **`BranchFamily`**:
  - Шаблон `((u1_sign, t1_sign), (u2_sign, t2_sign))`, u1 ≠ u2
  - k=0..n_tau лучей (k=0 = чистая фаза-2, k=n_tau = чистая фаза-1)
  - Вычисления векторизованы через numpy (нет Python-цикла по лучам)
  - grid: iso-offset линии (j dtau после переключения)
  - root_line, envelope, fan_surface
- 🆕 `src/core/surface_manager.py` — фабрика `ScalableFanSurface`, dh=0.002 (z-fighting)
- 🔄 `src/core/scalable_surface.py` — добавлен класс `ScalableFanSurface` (веер треугольников)
- 🔄 `config/colors.json` — добавлены `surface_plus_u`, `surface_minus_u`
- 🔄 `main.py` — n_u убран из params; два бранча `branch_plus/minus` с шаблонами `(+u,+t)→(-u,+t)` и `(-u,+t)→(+u,+t)`

**Технические детали:**
- `BranchFamily` независима от любой семьи — вычисляет boundary-узлы сама от root_spore
- Граница поверхности: `positions[0,:]` + `positions[1:-1,-1]` + `positions[-1,::-1]` = 3*n_tau-1 точек
- `SurfaceManager._count * dh` — offset по y для каждой поверхности

**Незакрытый баг (исправлено сессия 14):**
- `ScalableFanSurface` рендерит только линию от рута к одной угловой точке вместо полного веера. Скорее всего проблема в инициализации или порядке вершин треугольников.

**Участники:** Пользователь + Claude Sonnet 4.6

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

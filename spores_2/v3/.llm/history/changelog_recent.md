# Changelog - Последние сессии

**Last updated:** 2026-06-12 (v3 сессия 2 / общая сессия 16)

> Хранит последние 3 сессии. Если сессий стало > 3 — самую старую перенести в конец [changelog_archive.md](changelog_archive.md)
> Полная история → [changelog_archive.md](changelog_archive.md)

---

## 2026-06-12 (v3 сессия 2 / сессия 16) - Pendulum + мягкая радуга + arrow_scale

**Что сделано:**
- 🆕 `src/math/pendulum.py` — класс `Pendulum`: RK4 (n_sub=10) для `θ̈ = -sin(θ) + u`, `θ=0` — нижнее устойчивое положение, нормировка g/l=1, m·l²=1; векторизован (np.sin поэлементно), интерфейс `step(x0,v0,u,dt) -> (θ,ω)`
- 🔄 `src/math/double_integrator.py`, `src/math/__init__.py` — `DoubleIntegrator.step()` теперь возвращает `(x,v)` (был `np.array`), убран неиспользуемый `import numpy` — единый интерфейс с `Pendulum`
- 🔄 `src/spores/boundary_ray_family.py` — `_recompute()` зовёт `self._ctx.model.step(x, v, u, dt)` вместо инлайн-формулы double integrator; убран прямой импорт `_step`
- 🔄 `main.py` — новая секция `DYNAMICS MODEL`: `model = Pendulum(shared_context)`, `shared_context.bind('model', ...)` — переключение динамики = одна строка
- 🔄 `config/colors.json` — новая палитра "мягкая радуга": `node/edge/surface_{ppp..mmm}` — 8 цветов равномерно по кругу оттенков (шаг 45°, S≈0.55, V≈0.82), единый RGB на ветку с alpha 1.0/0.6/0.4; 4 цвета фазовых рёбер (`edge_fwd/bwd_pos/neg`) приведены к той же тональности
- 🆕 `src/core/scalable_arrow.py` — `size_factor` (default 1.0): `size = length/4 * size_factor`
- 🔄 `src/core/line_manager.py` — `arrow_scale` + `increase/decrease_arrow_size()`, применяется в `create_arrow()` и ко всем существующим стрелкам
- 🔄 `main.py` — `_resize()`: клавиша `1` (scroll/Q-E) меняет `spore_manager` и `line_manager.arrow_scale` одновременно
- 🔄 `src/core/input_manager.py` — Q/E дублируют логику `scroll up/down` (held-keys 1-4 приоритетнее зума)

**Технические детали:**
- Маятник: `θ̈ = sin(θ) + u` соответствует θ=0=верх (неустойчиво); для θ=0=низ (устойчиво) знак гравитационного члена меняется на `-sin(θ)`
- Класс был изначально `InvertedPendulum`, переименован в `Pendulum` после смены конвенции (при θ=0=низ это уже не "перевёрнутый" маятник)
- `model.step()` — общий интерфейс для `DoubleIntegrator`/`Pendulum`, диспетчеризуется через `shared_context.model`

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-05-27 (v3 сессия 1 / сессия 15) - ScalableArrow + семантические цвета + docs

**Что сделано:**
- 🆕 `src/core/scalable_arrow.py` — класс `ScalableArrow` (наследник ScalableLine): равнобедренный треугольник (leg=2×base) в середине линии, центр масс = середина сегмента; `t_sign` управляет направлением; `_tri_entity` всегда alpha=1; `__setattr__` синхронизирует color/enabled
- 🔄 `src/core/line_manager.py` — метод `create_arrow()`, alpha=1 принудительно для линии и треугольника
- 🔄 `src/spores/boundary_ray_family.py`:
  - `_BranchRay` — per-segment цвет и t_sign (switch_k определяет фазу каждого сегмента)
  - `BranchFamily._phase_edge()` — получает семантический цвет по (u_sign, t_sign)
  - Root lines → `create_arrow` с корректным t_sign
  - Grid линии **удалены** (упрощение визуализации)
  - Поверхность y=0.01, линии/узлы y=0.03, треугольники y=0.05 (разделённые слои)
- 🔄 `config/colors.json` — 4 семантических цвета: `edge_fwd_pos` (красный), `edge_fwd_neg` (синий), `edge_bwd_pos` (оранжевый), `edge_bwd_neg` (маджента); node_* alpha→1.0; surface_* alpha→0.5
- 🆕 `docs/reachability.md` — алгоритм ветвей, связь с принципом максимума Понтрягина, прямое/обратное достижимые множества, таблица терминов

**Технические детали:**
- Проблема прозрачности треугольников: `Color` в Ursina содержит alpha → при `_tri_entity.color = color` alpha перезаписывалась. Решение: `tri.alpha = 1.0` после синхронизации цвета в `__setattr__`
- Фаза сегмента j в луче с switch_k: фаза 1 если `j < switch_k - 1`, иначе фаза 2

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-05-26 (сессия 14) - Фикс ScalableFanSurface + BranchFamilyManager + 8 шаблонов

**Что сделано:**
- 🔄 `src/core/scalable_surface.py` — `ScalableFanSurface` полностью переписан: grid-триангуляция `positions[k,step]` вместо fan по boundary. Причина: boundary не была star-shaped относительно рута, что давало перекрытия треугольников и пятно. Подробнее → Decision #15
- 🔄 `src/core/surface_manager.py` — API: `n_pts` → `n_tau`, `curve_pts` → `positions_grid`
- 🔄 `src/spores/boundary_ray_family.py` — dirty flag (пересчёт только при изменении root_pos/a_max/tau); `_root_line2` (рут → positions[-1,0]); узлы: `GhostSpore` → `Spore` (GhostSpore.tick() перетирал real_position)
- 🆕 `src/spores/branch_family_manager.py` — фабрика `BranchFamilyManager`, авто-регистрирует tickable при создании
- 🔄 `main.py` — все 8 шаблонов через `BranchFamilyManager`
- 🔄 `config/colors.json` — 8 цветовых наборов: `ppp` (зелёный), `ppm` (бирюзовый), `pmp` (голубой), `pmm` (фиолетовый), `mpp` (оранжевый), `mpm` (розовый), `mmp` (золотой), `mmm` (лососевый)

**Технические детали:**
- Grid-триангуляция: fan (root → col[0]) + квады между соседними лучами. `n_tau + 2*n_tau*(n_tau-1)` треугольников, нет перекрытий
- Dirty flag: кешируем `root_pos.copy()`, `a_max`, `tau`; `np.array_equal` для сравнения
- Суффикс `{u1}{t1}{t2}` (p=+1, m=−1): полностью кодирует шаблон

**Участники:** Пользователь + Claude Sonnet 4.6

---

## Шаблон для новых записей

```markdown
## YYYY-MM-DD (сессия N) - Краткая тема

**Что сделано:**
- 🆕/🔄/🐛/🗑️ `файл` — что изменилось

**Участники:** Пользователь + Claude X
```

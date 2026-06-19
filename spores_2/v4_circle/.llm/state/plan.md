# Plan & Progress - План и прогресс

**Last updated:** 2026-06-20 (v4_circle сессия 2 / общая сессия 21)

---

## 🎯 Текущая задача

*(нет активной задачи — обсуждение дальнейшего развития CirclePattern)*

---

## ✅ Сделано в v4_circle сессии 1 (сессия 20)

- `src/spores/circle_pattern.py` — CirclePattern: круг спор вокруг look_point, 3 группы стрелок фазовой производной (u=-a_max/0/+a_max), траектории интегрирования из каждой споры
- `src/core/scalable_tip_arrow.py` — ScalableTipArrow: наконечник на конце стрелки
- `src/math/pendulum.py`, `src/math/double_integrator.py` — `derivative(x, v, u)` как отдельный метод
- `src/core/line_manager.py` — `create_tip_arrow()`
- `main.py` — v4_circle: параметры 1-6, ctrl+1/2/3 toggle групп
- `config/colors.json` — цвета circle/node, circle/edge, circle/arrow_neg/zero/pos

---

## ✅ Сделано в v3 сессии 5 (сессия 19)

- `main.py`, `src/core/input_manager.py` — toggle видимости перенесён с `1`-`8` на `control+1`-`control+8`

---

## ✅ Сделано в v3 сессии 4 (сессия 18)

- `src/spores/boundary_ray_family.py` — `_BranchRay.set_visible()`, `BranchFamily.set_visible()/toggle()`; `_visible` сохраняется через rebuild (`_build()` зовёт `set_visible(self._visible)`)
- `src/core/line_manager.py` — `set_visible(name, visible)`
- `src/core/surface_manager.py` — `toggle(name)`, `set_visible(name, visible)`
- `main.py` — все 8 веток присвоены переменным (`b_ppp` ... `b_mmm`); клавиши `1`-`8` (press) toggle'ят видимость соответствующей ветки

---

## ✅ Сделано в v3 сессии 2 (сессия 16)

- `src/math/pendulum.py` — новый класс `Pendulum`: RK4 (n_sub=10), `θ̈ = -sin(θ) + u`, `θ=0` — нижнее устойчивое положение (изначально был `InvertedPendulum` с `θ=0`=верх, переименован и сменена конвенция по запросу)
- `src/math/double_integrator.py`, `__init__.py` — единый интерфейс `step(x0,v0,u,dt) -> (x,v)` для обеих моделей
- `src/spores/boundary_ray_family.py` — `_recompute()` зовёт `ctx.model.step(...)` вместо инлайн-формулы double integrator
- `main.py` — секция `DYNAMICS MODEL`, `model = Pendulum(shared_context)`, `shared_context.bind('model', ...)`
- `config/colors.json` — новая палитра "мягкая радуга": 8 цветов node/edge/surface_{ppp..mmm} равномерно по кругу оттенков (S≈0.55, V≈0.82), 4 семантических цвета фазовых рёбер приведены к той же тональности
- `src/core/scalable_arrow.py` — `size_factor` (множитель размера наконечника)
- `src/core/line_manager.py` — `arrow_scale` + `increase/decrease_arrow_size()`
- `main.py` — `_resize()`: клавиша `1`+scroll/Q-E меняет размер спор и стрелок одновременно
- `src/core/input_manager.py` — Q/E дублируют логику scroll up/down (параметры 1-4 приоритетнее зума)

---

## ✅ Сделано в v3 сессии 1 (сессия 15)

- `src/core/scalable_arrow.py` — новый класс ScalableArrow, равнобедренный треугольник в середине, t_sign управляет направлением
- `src/core/line_manager.py` — метод `create_arrow()`, принудительный alpha=1
- `src/spores/boundary_ray_family.py` — per-segment цвет и t_sign (_BranchRay), _phase_edge(), root lines со стрелками, grid линии удалены, разделённые Y для поверхности и остального
- `config/colors.json` — 4 семантических цвета рёбер, node alpha=1, surface alpha=0.5
- `docs/reachability.md` — документация алгоритма + теория достижимости

---

## ✅ Сделано в сессии 14

- `scalable_surface.py` — `ScalableFanSurface` полностью переписан: grid-триангуляция `positions[k,step]` вместо fan по boundary
- `surface_manager.py` — API изменён: `n_pts` → `n_tau`; `curve_pts` → `positions_grid`
- `boundary_ray_family.py` — dirty flag (пересчёт только при изменении root/a_max/tau); `_root_line2` (рут → positions[-1,0]); узлы: GhostSpore → Spore
- `branch_family_manager.py` — новый менеджер-фабрика, авто-регистрирует tickable
- `main.py` — все 8 шаблонов через `BranchFamilyManager`
- `config/colors.json` — 8-цветная палитра (ppp..mmm)

## ✅ Сделано в сессии 13

- `scalable_surface.py` — `mode='ngon'` вместо `mode='triangle'` + убраны явные tris
- `surface_manager.py` — добавлен `double_sided=True`
- `boundary_ray_family.py` — boundary переписан чище: `positions[0,:]` + `positions[1:, n-1]` + `positions[-1,-2::-1]`
- `tests/test_boundary.ipynb` — ноутбук для визуализации boundary (matplotlib); подтверждено что boundary корректен

---

## ✅ Сделано в сессии 12

- `config/monitors.json` — создан, WindowManager читает его; margin для left монитора
- `WindowManager` — убран хардкод MONITORS, добавлен `get_margin()`
- `ghost_spore_family.py` — удалён целиком
- `boundary_ray_family.py` — переписан как самостоятельный `BranchFamily`:
  - k=0..n_tau лучей, шаблон управления, numpy-векторизация
  - grid iso-offset линии, root_line, envelope, fan_surface
- `surface_manager.py` — новый (фабрика ScalableFanSurface)
- `scalable_surface.py` — добавлен `ScalableFanSurface`
- `colors.json` — surface_plus_u / surface_minus_u

---

## 📅 Краткосрочные цели

### 1. ~~Починить ScalableFanSurface~~ ✅ ГОТОВО (сессия 14)

### 2. ~~Git коммит v3~~ ✅ ГОТОВО (сессия 16)

---

## 🗓️ Среднесрочные цели

### Векторизация на GPU (cupy)
- Текущие вычисления уже numpy-векторизованы
- При больших n_tau (100+) заменить numpy → cupy (drop-in)
- **Приоритет:** Низкий

### BranchFamily второго поколения
- Ветки из крайних точек существующей ветки
- Архитектурно: BranchFamily принимает другой BranchFamily как source вместо GhostSpore

---

## 🔮 Долгосрочные цели

- ~~Нелинейная динамика вместо DoubleIntegrator~~ ✅ ГОТОВО (сессия 16, `Pendulum`)
- Юнит-тесты для математики

---

## 🎨 Backlog

- Визуализация invariant point на полу
- Grid с масштабируемыми числами
- Сохранение/загрузка позиции камеры

# Current State - Текущее состояние проекта

**Last updated:** 2026-06-12 (v3 сессия 4 / общая сессия 18)

---

## ✅ Что работает

### Базовая функциональность:
- ✅ **Камера** — WASD, Mouse look, Space/Shift, Alt курсор
- ✅ **Zoom система** — scroll/Q/E zoom, R reset, математически корректная с invariant point
- ✅ **ScreenManager + Message** — динамический текст, UI-отступы читаются из monitors.json

### Конфиги (сессия 12):
- ✅ **`config/monitors.json`** — настройки мониторов (size, position, margin); WindowManager читает его
- ✅ **`config/colors.json`** — цвета включая `surface_plus_u/minus_u`
- ✅ **`WindowManager.get_margin()`** — возвращает (mx, my) для текущего монитора

### BranchFamily (сессии 12–15 / v3):
- ✅ **`BranchFamily`** в `src/spores/boundary_ray_family.py`
  - Шаблон `((u1_sign, t1_sign), (u2_sign, t2_sign))`, u1 ≠ u2
  - k=0..n_tau лучей (k=0 = чистая фаза-2, k=n_tau = чистая фаза-1)
  - Numpy-векторизация (нет Python-цикла по лучам)
  - root lines + envelope — со стрелками, grid линии **удалены**
  - **Dirty flag**: _recompute вызывается только при изменении root_pos / a_max / tau
  - Высоты: поверхность y=0.01, линии/точки y=0.03, треугольники стрелок y=0.05
- ✅ **`BranchFamilyManager`** в `src/spores/branch_family_manager.py`
- ✅ **`SurfaceManager`** + **`ScalableFanSurface`** — grid-триангуляция, alpha=0.5
- ✅ **Все 8 шаблонов** активны в main.py

### Видимость веток (v3 сессия 4 / сессия 18):
- ✅ Клавиши `1`-`8` (press) — toggle видимости каждой из 8 веток целиком (споры, рёбра-стрелки, root-линии, envelope, fan_surface)
  - `BranchFamily.set_visible()/toggle()` — управляет всеми визуальными элементами ветки
  - Новые методы: `_BranchRay.set_visible()`, `LineManager.set_visible()`, `SurfaceManager.set_visible()/toggle()`
  - Состояние `_visible` сохраняется через rebuild (`_build()` вызывает `set_visible(self._visible)` в конце) — переживает смену n_tau
  - Не конфликтует с клавишами `1`-`4` (mode='scroll') для resize/tau/a_max/n_tau — toggle срабатывает на press, параметры на held+scroll

### ScalableArrow (v3 сессия 1, доработано сессия 16):
- ✅ **`ScalableArrow`** в `src/core/scalable_arrow.py` — наследуется от ScalableLine
  - Равнобедренный треугольник (leg = 2×base) в середине линии, центр масс = середина
  - `t_sign` управляет направлением: +1 = p1→p2, −1 = p2→p1
  - Отдельный `_tri_entity`, всегда alpha=1 (непрозрачный)
  - `__setattr__` синхронизирует color/enabled на треугольник
  - `size_factor` (по умолчанию 1.0) — множитель размера наконечника: `size = length/4 * size_factor`
- ✅ **`LineManager.create_arrow()`** — создаёт ScalableArrow, alpha=1 принудительно, применяет `arrow_scale`
- ✅ **`LineManager.increase/decrease_arrow_size()`** — меняют `arrow_scale` для всех стрелок + `update_transform()`

### Цветовая схема v3 (палитра обновлена сессия 16):
- ✅ "Мягкая радуга" — `node/edge/surface_{ppp..mmm}`: 8 цветов равномерно по кругу оттенков (шаг 45°), S≈0.55 V≈0.82, единый RGB на ветку
  - alpha: node=1.0, edge=0.6, surface=0.4
- ✅ 4 семантических цвета фазовых рёбер по (t_sign, u_sign) — `edge_fwd_pos/neg`, `edge_bwd_pos/neg` — приведены к той же тональности (коралл/голубой/персик/розовый)

### Документация:
- ✅ **`docs/reachability.md`** — алгоритм ветвей, связь с принципом Понтрягина, термины

### Параметры (актуальное):
- `1` + scroll/Q-E — spore size + arrow size (синхронно, `_resize()` в main.py)
- `2` + scroll/Q-E — tau
- `3` + scroll/Q-E — a_max (min=0)
- `4` + scroll/Q-E — n_tau (шаг 1, min=0)
- `scroll`/`Q-E` — zoom (если не зажата клавиша параметра 1-4)
- *(n_u убран — концепция семьи удалена)*

### Математика (модель — сессия 16):
- ✅ `DoubleIntegrator` — 1D, stateless `step(x0,v0,u,dt) -> (x,v)`, a_max из SharedContext
- ✅ `Pendulum` (`src/math/pendulum.py`) — нелинейный маятник, RK4 (n_sub=10)
  - `θ=0` — нижнее устойчивое положение, `θ̈ = -sin(θ) + u`, нормировка g/l=1, m·l²=1
  - Векторизован (np.sin поэлементно), интерфейс step() идентичен DoubleIntegrator
- ✅ Модель подключается через `shared_context.model` — `BranchFamily._recompute()` зовёт `ctx.model.step(...)`
  - Переключение модели = одна строка в `main.py` (`model = Pendulum(shared_context)`)

---

## ❌ Что сломано

*(критических багов нет)*

---

## 📁 Структура src/

```
src/
  core/
    color_manager.py
    line_manager.py
    surface_manager.py    ← НОВЫЙ (сессия 12)
    window_manager.py     ← читает monitors.json
    input_manager.py
    update_manager.py
    scene_manager.py
    shared_context.py
    param_manager.py
    object_manager.py
    scalable.py
    scalable_line.py
    scalable_arrow.py     ← НОВЫЙ (v3 сессия 1)
    scalable_surface.py   ← добавлен ScalableFanSurface
    screen_manager.py
    zoom_manager.py

  spores/
    spore.py
    spore_manager.py
    boundary_ray_family.py   ← BranchFamily
    branch_family_manager.py ← НОВЫЙ (сессия 14)
    *(ghost_spore_family.py удалён)*

  math/
    double_integrator.py
    pendulum.py            ← НОВЫЙ (сессия 16)
    spore_integrator.py
```

---

## 🚀 Как запустить

```bash
python main.py        # обычный запуск (тест: НЕ через run.py!)
python run.py         # с автоперезапуском (разработка)
```

---

**Статус:** 🟢 Работает

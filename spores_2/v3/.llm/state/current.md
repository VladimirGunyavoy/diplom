# Current State - Текущее состояние проекта

**Last updated:** 2026-05-27 (v3 сессия 1 / общая сессия 15)

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

### ScalableArrow (v3 сессия 1):
- ✅ **`ScalableArrow`** в `src/core/scalable_arrow.py` — наследуется от ScalableLine
  - Равнобедренный треугольник (leg = 2×base) в середине линии, центр масс = середина
  - `t_sign` управляет направлением: +1 = p1→p2, −1 = p2→p1
  - Отдельный `_tri_entity`, всегда alpha=1 (непрозрачный)
  - `__setattr__` синхронизирует color/enabled на треугольник
- ✅ **`LineManager.create_arrow()`** — создаёт ScalableArrow, alpha=1 принудительно

### Цветовая схема v3:
- ✅ 4 семантических цвета рёбер по (t_sign, u_sign):
  - `edge_fwd_pos` — красный (t>0, u>0)
  - `edge_fwd_neg` — синий (t>0, u<0)
  - `edge_bwd_pos` — оранжевый (t<0, u>0)
  - `edge_bwd_neg` — маджента/фиолетовый (t<0, u<0)
- ✅ Цвета узлов (node_*) — alpha=1.0, непрозрачные
- ✅ Цвета поверхностей (surface_*) — alpha=0.5

### Документация:
- ✅ **`docs/reachability.md`** — алгоритм ветвей, связь с принципом Понтрягина, термины

### Параметры (актуальное):
- `1` + scroll — spore size
- `2` + scroll — tau
- `3` + scroll — a_max (min=0)
- `4` + scroll — n_tau (шаг 1, min=0)
- `scroll` — zoom
- *(n_u убран — концепция семьи удалена)*

### Математика:
- ✅ `DoubleIntegrator` — 1D, stateless step, a_max из SharedContext
- ✅ `_step()` — numpy, JIT-готова

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

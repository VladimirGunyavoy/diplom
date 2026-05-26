# Current State - Текущее состояние проекта

**Last updated:** 2026-05-26 (сессия 14)

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

### BranchFamily (сессии 12–14):
- ✅ **`BranchFamily`** в `src/spores/boundary_ray_family.py` — самостоятельная, независима от семьи
  - Шаблон `((u1_sign, t1_sign), (u2_sign, t2_sign))`, u1 ≠ u2
  - k=0..n_tau лучей (k=0 = чистая фаза-2, k=n_tau = чистая фаза-1)
  - Numpy-векторизация (нет Python-цикла по лучам)
  - root_line + root_line2: рут → positions[0,0] и рут → positions[-1,0]
  - envelope: полилиния через крайние точки (n_tau сегментов)
  - grid: iso-offset линии (j dtau после переключения для каждого j)
  - **Dirty flag**: _recompute вызывается только при изменении root_pos / a_max / tau
- ✅ **`BranchFamilyManager`** в `src/spores/branch_family_manager.py` — фабрика, авто-регистрирует tickable
- ✅ **`SurfaceManager`** в `src/core/surface_manager.py` — фабрика ScalableFanSurface, z-fighting offset
- ✅ **`ScalableFanSurface`** в `src/core/scalable_surface.py` — grid-триангуляция, zoom-трансформ работает
  - fan из рута к первому столбцу + grid квады между соседними лучами
  - **НЕТ перекрытий** (граница не была star-shaped → заменена прямой сеткой)
- ✅ **Все 8 шаблонов** активны в main.py, 8-цветная палитра в colors.json

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

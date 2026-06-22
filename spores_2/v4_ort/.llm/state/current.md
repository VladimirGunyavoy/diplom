# Current State - Текущее состояние проекта

**Last updated:** 2026-06-23 (v4_ort сессия 2 / общая сессия 21)

---

## ✅ Что работает

### Базовая функциональность (из v3):
- ✅ **Камера** — WASD, Mouse look, Space/Shift, Alt курсор
- ✅ **Zoom система** — scroll/Q/E zoom, R reset, математически корректная с invariant point
- ✅ **ScreenManager + Message** — динамический текст, UI-отступы из monitors.json

### Конфиги:
- ✅ **`config/monitors.json`** — настройки мониторов; WindowManager читает его
- ✅ **`config/colors.json`** — цвета, 4 семантических цвета рёбер по (t_sign, u_sign)

### OrthoGrid (v4_ort сессия 1 / сессия 20):
- ✅ **`OrthoGrid`** в `src/spores/ortho_grid.py`
  - Flow-orthogonal сетка в фазовом пространстве для фиксированного u
  - Time axis: интегрируем look_point вперёд/назад n_tau шагов
  - Ortho axis: шаг ⊥ фазовой скорости, затем интеграция
  - Три режима шага клонов: **TIME** (одинаковый dtau), **ARC** (match arc length центральной), **ORTHO** (optimize front ⊥ к вектору поля)
  - Dirty flag: пересчёт только при изменении look_point/a_max/tau/r_s
  - Front lines на каждом time step между соседними траекториями
  - Surfaces: fwd и bwd половины (grid-триангуляция)
  - Alpha fade для крайних траекторий
- ✅ **Два экземпляра** `grid_p` (+1) и `grid_m` (-1) — переключение u_sign клавишей 8 (цикл +1/-1/±1)
- ✅ **`set_active()`** — управляет видимостью всех элементов grid
- ✅ **`cycle_mode()`** — переключение TIME/ARC/ORTHO клавишей 7

### Динамика:
- ✅ `DoubleIntegrator` — 1D, `step(x0,v0,u,dt) -> (x,v)`, `deriv(x,v,u) -> (fx,fv)`
- ✅ `Pendulum` — RK4, `θ̈ = -sin(θ) + u`, `deriv()` для ⊥-шага
- ✅ Модель подключается через `shared_context.model`

### ScalableArrow:
- ✅ Равнобедренный треугольник в середине линии, t_sign управляет направлением
- ✅ `size_factor`, `arrow_scale` в LineManager

### Параметры (актуальное):
- `1` + scroll/Q-E — spore size
- `2` + scroll/Q-E — a_max
- `3` + scroll/Q-E — tau
- `4` + scroll/Q-E — n_tau (шаг 1, min=0)
- `5` + scroll/Q-E — n_s (число ⊥-шагов, шаг 1, min=0)
- `6` + scroll/Q-E — r_s (разброс ⊥, exp)
- `7` (press) — cycle mode (TIME/ARC/ORTHO)
- `8` (press) — cycle u_sign (+1/-1/±1)
- `scroll`/`Q-E` — zoom (если не зажата клавиша параметра)

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
    surface_manager.py
    window_manager.py
    input_manager.py
    update_manager.py
    scene_manager.py
    shared_context.py
    param_manager.py
    object_manager.py
    scalable.py
    scalable_line.py
    scalable_arrow.py
    scalable_surface.py
    screen_manager.py
    zoom_manager.py
    frame.py

  spores/
    spore.py
    spore_manager.py
    ortho_grid.py       ← НОВЫЙ (v4_ort s1)
    trajectories.py

  math/
    double_integrator.py
    pendulum.py
    spore_integrator.py

  utils/
    nb_logger.py
```

---

## 🚀 Как запустить

```bash
python main.py        # обычный запуск
python run.py         # с автоперезапуском (разработка)
```

---

**Статус:** 🟢 Работает

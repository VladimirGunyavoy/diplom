# Current State - Текущее состояние проекта

**Last updated:** 2026-04-24 (сессия 11)

---

## ✅ Что работает

### Базовая функциональность:
- ✅ **Камера** (FirstPersonController) — WASD, Mouse look, Space/Shift, Alt курсор
- ✅ **Zoom система** — scroll/Q/E zoom, R reset, математически корректная с invariant point
- ✅ **Визуализация** — Frame (XYZ оси), клетчатый пол, Spore (Circle), GhostSpore
- ✅ **ScreenManager + Message** — динамический текст на экране через getter каждый кадр

### Архитектура (сессия 11):
- ✅ **`config/colors.json`** — создан, ColorManager читает его (путь исправлен)
- ✅ **`LineManager`** — фабрика ScalableLine объектов, регистрирует в ZoomManager
- ✅ **`GhostSporeFamily`** — рёбра графа (time + control + root→gen1), параметры `name`/`time_sign`/`color_key`
- ✅ **`_GhostLineFamily`** — приватный класс внутри ghost_spore_family.py
- ✅ **`family_b`** — вторая семья с `time_sign=-1` (обратное время)
- ✅ **`SharedContext`** — менеджеры как прямые поля (`ctx.line_manager`, `ctx.color_manager`, etc.)
- ✅ **Граничные ноды** — j=±n_u подсвечены (зелёный/красный)
- ✅ **`BoundaryRay`** — луч из граничной точки с противоположным управлением
- ✅ **`BoundaryRayFamily`** — семья лучей из одной границы (plus_u / minus_u)
- ✅ **`SporeManager.register`** — новые споры получают текущий `size` при ребилде

### Параметры (актуальное):
- `1` + scroll — spore size
- `2` + scroll — tau
- `3` + scroll — a_max (min=0)
- `4` + scroll — n_tau (шаг 1, min=0)
- `5` + scroll — n_u (шаг 1, min=0)
- `scroll` — zoom

### Математика (src/math/):
- ✅ `DoubleIntegrator` — 1D, stateless step, a_max из SharedContext
- ✅ `SporeIntegrator` — 2D unicycle model (x, y, theta)

---

## 🔄 Что в процессе

- 🔄 **BoundaryRayFamily** — работает, но визуально "каша" при включённых двух семьях + лучах одновременно. Нужно переключение видимости или настройка.

---

## ❌ Что сломано / не работает

*(нет критических проблем)*

---

## 📁 Структура src/

```
src/
  core/
    color_manager.py       ← читает config/colors.json
    line_manager.py        ← НОВЫЙ: фабрика ScalableLine
    window_manager.py
    input_manager.py
    update_manager.py
    scene_manager.py
    shared_context.py      ← менеджеры как прямые поля
    param_manager.py
    object_manager.py
    scalable.py
    scalable_line.py
    screen_manager.py
    zoom_manager.py

  spores/
    spore.py
    spore_manager.py
    ghost_spore_family.py  ← рёбра + name/time_sign/color_key + _GhostLineFamily
    boundary_ray_family.py ← НОВЫЙ: BoundaryRay + BoundaryRayFamily

  math/
    double_integrator.py
    spore_integrator.py
```

---

## 🚀 Как запустить

```bash
python main.py        # обычный запуск
python run.py         # с автоперезапуском
```

---

**Статус:** 🟢 Стабилен

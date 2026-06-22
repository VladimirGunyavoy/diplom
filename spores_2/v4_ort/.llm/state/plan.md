# Plan & Progress - План и прогресс

**Last updated:** 2026-06-23 (v4_ort сессия 2 / общая сессия 21)

---

## 🎯 Текущая задача

*(нет активной задачи)*

---

## ✅ Сделано в v4_ort сессии 1 (сессия 20)

- 🆕 `src/spores/ortho_grid.py` — класс `OrthoGrid`: flow-orthogonal сетка в фазовом пространстве
  - Три режима шага клонов: TIME (одинаковый dtau), ARC (match arc length центральной), ORTHO (optimize front ⊥ через minimize_scalar)
  - `_ortho_step()` — шаг ⊥ фазовой скорости с fallback вблизи equilibria
  - `_integrate_by_arc()` — интеграция с matching arc length
  - `_integrate_by_ortho()` — оптимизация ⊥ фронтов
  - Front lines между соседними траекториями на каждом time step
  - Surfaces fwd/bwd (grid-триангуляция)
  - `set_active()`, `cycle_mode()`
- 🔄 `src/math/double_integrator.py`, `src/math/pendulum.py` — добавлен `deriv(x,v,u) -> (fx,fv)` в обе модели
- 🔄 `main.py` — два экземпляра OrthoGrid (±u_sign), _cycle_u(), параметры n_s и r_s, клавиши 5-8
- 🔄 `src/core/input_manager.py` — поддержка новых биндингов

---

## ✅ Предыдущие сессии (v3)

- сессия 18: toggle видимости веток (клавиши 1-8)
- сессия 17: миграция .llm протокола v1 → v2
- сессия 16: Pendulum + мягкая радуга + arrow_scale
- сессия 15: ScalableArrow + семантические цвета + docs
- сессия 14: фикс ScalableFanSurface + BranchFamilyManager + 8 шаблонов

---

## 📅 Краткосрочные цели

*(определяются пользователем)*

---

## 🗓️ Среднесрочные цели

### Векторизация на GPU (cupy)
- При больших n_tau/n_s заменить numpy → cupy
- **Приоритет:** Низкий

---

## 🔮 Долгосрочные цели

- Юнит-тесты для математики

---

## 🎨 Backlog

- Визуализация invariant point на полу
- Grid с масштабируемыми числами
- Сохранение/загрузка позиции камеры

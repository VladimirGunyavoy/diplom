# Plan & Progress - План и прогресс

**Last updated:** 2026-05-26 (сессия 14)

---

## 🎯 Текущая задача

*(нет активной задачи)*

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

### 2. Git коммит сессий 11–14
**Статус:** Не сделан

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

- Нелинейная динамика вместо DoubleIntegrator
- Юнит-тесты для математики

---

## 🎨 Backlog

- Визуализация invariant point на полу
- Grid с масштабируемыми числами
- Сохранение/загрузка позиции камеры

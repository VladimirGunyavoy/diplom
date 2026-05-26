# Changelog - Последние сессии

**Last updated:** 2026-05-27 (v3 сессия 1 / общая сессия 15)

> Хранит последние 3 сессии. Если сессий стало > 3 — самую старую перенести в конец [changelog_archive.md](changelog_archive.md)
> Полная история → [changelog_archive.md](changelog_archive.md)

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

**Незакрытый баг:**
- `ScalableFanSurface` рендерит только линию от рута к одной угловой точке вместо полного вееpa. Скорее всего проблема в инициализации или порядке вершин треугольников. **Следующей сессии — починить.**

**Участники:** Пользователь + Claude Sonnet 4.6

---

## Шаблон для новых записей

```markdown
## YYYY-MM-DD (сессия N) - Краткая тема

**Что сделано:**
- 🆕/🔄/🐛/🗑️ `файл` — что изменилось

**Участники:** Пользователь + Claude X
```

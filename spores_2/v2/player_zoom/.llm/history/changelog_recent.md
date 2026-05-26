# Changelog - Последние сессии

**Last updated:** 2026-05-26 (сессия 14)

> Хранит последние 3 сессии. Если сессий стало > 3 — самую старую перенести в конец [changelog_archive.md](changelog_archive.md)
> Полная история → [changelog_archive.md](changelog_archive.md)

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

## 2026-04-24 (сессия 11) - Рёбра графа + BoundaryRayFamily

**Что сделано:**
- 🐛 `run.py`, `watcher.py` — фикс путей (PROJECT_ROOT указывал на `src/` вместо `player_zoom/`)
- 🐛 `color_manager.py` — фикс пути к colors.json (тот же баг `..` → `../..`)
- 🆕 `config/colors.json` — создан со всеми цветами (frame, scene, family, family_b, boundary, ray)
- 🆕 `src/core/line_manager.py` — фабрика `ScalableLine`, регистрирует напрямую в ZoomManager
- 🔄 `ghost_spore_family.py` — рёбра time/control/root→gen1 через `_GhostLineFamily`; параметры `name`, `time_sign`, `color_key`; граничные ноды j=±n_u подсвечены; свойства `nodes`, `n_tau`, `n_u`, `time_sign`
- 🆕 `src/spores/boundary_ray_family.py` — `BoundaryRay` (одна траектория) + `BoundaryRayFamily` (все лучи из одной границы)
- 🔄 `shared_context.py` — менеджеры как прямые поля (`ctx.color_manager`, `ctx.line_manager`, etc.)
- 🔄 `spore_manager.py` — новые споры получают текущий `size` при ребилде семьи
- 🔄 `zoom_manager.py`, `object_manager.py` — убраны print при каждом создании объекта
- 🔄 `main.py` — `family_b` (time_sign=-1), `ray_family_plus/minus`

**Технические детали:**
- `_GhostLineFamily` — приватный класс, живёт только внутри ghost_spore_family.py; `None` как маркер "источник = корень"
- `BoundaryRay.recompute(start_pos, u, dt, di)` — чистый stateless пересчёт
- `BoundaryRayFamily.tick()` — проверяет изменение n_tau/n_u у родительской семьи, синхронизируется

**Известная проблема:**
- Визуально "каша" при одновременном показе family_a + family_b + лучей. Нужно переключение видимости (следующая задача).

**Участники:** Пользователь + Claude Sonnet 4.6

---

## Шаблон для новых записей

```markdown
## YYYY-MM-DD (сессия N) - Краткая тема

**Что сделано:**
- 🆕/🔄/🐛/🗑️ `файл` — что изменилось

**Участники:** Пользователь + Claude X
```

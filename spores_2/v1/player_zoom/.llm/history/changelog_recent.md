# Changelog - Последние сессии

**Last updated:** 2026-04-24

> Хранит последние 3 сессии. Если сессий стало > 3 — самую старую перенести в конец [changelog_archive.md](changelog_archive.md)
> Полная история → [changelog_archive.md](changelog_archive.md)

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

## 2026-04-23 (сессия 10) - GhostSporeFamily + архитектура менеджеров

**Что сделано:**
- 🔄 `SporeManager` — добавлен `create(cls, name, **kwargs)` как прокси к ObjectManager; убрана зависимость ObjectManager → SporeManager
- 🔄 `ObjectManager` — принимает `shared_context`, auto-inject `ctx` для Spore-субклассов, добавлен `register_tickable()` для не-GameObject объектов с tick()
- 🔄 `ParamManager` — добавлены `min_val`/`max_val` с clamping; новые параметры `a_max` (кл. 3), `n_tau` (кл. 4), `n_u` (кл. 5)
- 🔄 `SharedContext` — `param_manager` забиндан как единая точка доступа к параметрам
- 🔄 `DoubleIntegrator` — рефакторинг: stateless `step(x0, v0, u, t)`, принимает SharedContext, `tick()` синхронизирует `a_max`, убрано внутреннее состояние позиции
- 🆕 `src/spores/ghost_spore_family.py` — `GhostSporeFamily`: сетка `n_tau × (2*n_u+1)` призрачных спор, рекурсивная эволюция через DI

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-04-23 (сессия 9) - Рефакторинг архитектуры: tick/register/ParamManager/структура src/

**Что сделано:**
- 🔄 `register(**kwargs)` — универсальный метод регистрации вместо отдельных `register_X()` в InputManager и UpdateManager
- 🔄 `tick()` — унифицированное имя per-frame метода у всех компонентов
- 🆕 `src/core/param_manager.py` — `ParamManager`: именованные float-параметры, exp/linear режимы
- 🗑️ `src/tau_manager.py` — удалён, tau теперь `param_manager.add('tau', 0.5)`
- 🔄 `src/core/input_manager.py` — `bind()` с mode='press' и mode='scroll', hold+scroll подавляет зум
- 🔄 `src/` реорганизована: `core/`, `spores/`, `math/`, `utils/`
- 🔄 `SceneSetup` → `SceneManager`

**Участники:** Пользователь + Claude Sonnet 4.6

---

## Шаблон для новых записей

```markdown
## YYYY-MM-DD (сессия N) - Краткая тема

**Что сделано:**
- 🆕/🔄/🐛/🗑️ `файл` — что изменилось

**Участники:** Пользователь + Claude X
```

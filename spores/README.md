# Проект "Споры" (версия 15 Tree)

> 🌱 Интерактивная 3D-симуляция для исследования алгоритмов управления и построения **деревьев достижимости** с поддержкой **обратной динамики**

![Статус проекта](https://img.shields.io/badge/status-development-yellow)
![Версия](https://img.shields.io/badge/version-v15_tree-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Лицензия](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Что нового в v15_tree?

### 👻 Предсказания и «призраки»
- **PredictionVisualizer** — генерация и отображение «призрачных» спор (ghost spores) под разные управления  
- **Ghost Links** — связи от активной споры к оптимальной «призрачной»  

### 🌳 Дерево спор
- **Алгоритмы пар/внуков** — поиск и оптимизация траекторий по парам спор  
- **Оптимизация площади дерева** (`area_opt`) — численные методы, ускоряемые `numba`  
- **Конвергенция** — таблицы встреч и хронология спор  

### 🧬 Новый SporeManager
- Упрощённая регистрация/удаление спор  
- Управление связями (Links) и предсказаниями  
- Интеграция с ZoomManager, ColorManager, AngelManager  

### ⚡ Производительность
- `numba` для тяжёлых мест  
- Профилировка (time/perf профайлеры)  
- Watcher для автоперезапуска демо  

---

## 🚀 Быстрый старт

### 1) Требования
- **Python 3.10+**
- **Git**

### 2) Установка
```bash
git clone <URL репозитория>
cd <repo_root>

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r v15_tree/requirements.txt
```

### 3) Запуск демо

```bash
python -m v15_tree.scripts.run.main_demo
```

### 4) Режим разработки

```bash
# 🔥 автоперезапуск при изменениях
python v15_tree/src/utils/watcher.py main_demo
```

---

## 🎮 Управление (hotkeys)

### Камера

* **WASD** — движение
* **Space / Shift** — вверх / вниз
* **Мышь** — поворот (при захваченном курсоре)
* **Alt** — захват/освобождение курсора
* **Q / Esc** — выход

### Споры

* **ЛКМ** — создать спору
* **F** — создать новую спору от последней активной
* **C** — очистить споры и историю

### Время

* **◀️ / ▶️** — шаг назад / вперёд
* **B** — переключить режим обратной динамики
* **M / N** — работа с временными метками

### Визуализация

* **Y** — траектории в прошлое
* **U** — временная шкала
* **H** — скрыть/показать UI

---

## 📁 Структура проекта

```
v15_tree/
  ├─ src/
  │   ├─ core/       # ядро симуляции (Spore, BackwardDynamics, StateHistory)
  │   ├─ logic/      # математика/алгоритмы (cost, optimizer, tree, pairs, area_opt)
  │   ├─ managers/   # SporeManager, PredictionManager, Time/Zoom/Color/Angel
  │   ├─ visual/     # визуализация (споры, призраки, ссылки, UI)
  │   └─ utils/      # watcher, отладка, утилиты
  ├─ scripts/
  │   └─ run/
  │       ├─ main_demo.py   # главный сценарий
  │       └─ tests/         # стенды оптимизации/топологии
  ├─ config/       # конфиги/пресеты
  ├─ docs/         # quick_start, trajectory_optimization и др.
  ├─ theory/       # теоретические заметки
  └─ requirements.txt
```

---

## 🧠 Архитектура v15\_tree

* **core/** — Spore, BackwardDynamics, история состояний
* **logic/** — cost\_function, optimizer, spawn\_area, tree (pairs, area\_opt)
* **managers/** — SporeManager, PredictionManager, Time/Zoom/Color/Angel
* **visual/** — `PredictionVisualizer`, `Link`, «ангелы», «пилоны», UI
* **utils/** — watcher, debug helpers, numba utils

---

## 📚 Документация

* **[🚀 Quick Start](v15_tree/docs/quick_start.md)**
* **[📐 Теория](v15_tree/theory/README.md)**
* **[⚡ Trajectory Optimization](v15_tree/docs/trajectory_optimization.md)**
* **[🔍 Best Practices](v15_tree/docs/best_practices.md)**

---

## 🔄 Миграция с v14\_back

* ✅ PredictionManager + ghost spores
* ✅ Дерево спор (pairs, area\_opt)
* ✅ Новый SporeManager
* ✅ Python 3.10+ и `numba`

---

## 📄 Лицензия

Проект распространяется под лицензией MIT. См. файл `LICENSE`.

---

<div align="center">

**⭐ Если проект вам полезен — поставьте звезду! ⭐**

*Создано с ❤️ для обратной динамики, деревьев достижимости и визуального анализа*

</div>

# 🧬 Проект "Споры" v16 — Manual Picker System

> 🎯 **Интерактивный пикинг и исследование спор под курсором**

Система симуляции эволюционных агентов с обратной динамикой маятника и новой системой интерактивного выбора спор.

![Версия](https://img.shields.io/badge/версия-v16_picker-blue)
![Фокус](https://img.shields.io/badge/фокус-manual_picker-green)
![Архитектура](https://img.shields.io/badge/архитектура-spatial_indexing-orange)

## 🎯 Новое в v16: Manual Picker System

### **Основная идея**
Пользователь может наводить курсор на споры и получать детальную информацию о них:
- 🔍 **Spatial Detection** — эффективный поиск спор под курсором
- 📊 **Spore Analysis** — исследование выбранной споры и её связей  
- 🌳 **Tree Exploration** — анализ детей и внуков споры
- 🎯 **Interactive Selection** — интуитивный интерфейс выбора

### **Ключевые компоненты v16**
- **ManualPicker** — основной класс для пикинга спор
- **SporeIndex** — эффективное пространственное хранение  
- **CursorDetection** — точное определение позиции курсора в мировых координатах
- **SporeAnalyzer** — извлечение информации о связях и иерархии

## 🏗️ Архитектурная схема v16

```mermaid
graph TB
    subgraph "🎯 v16: Manual Picker System"
        ManualPicker["`🎯 ManualPicker`<br/>Главный контроллер"]
        SporeIndex["`📍 SporeIndex`<br/>Пространственный поиск"]
        CursorDetector["`🖱️ CursorDetector`<br/>Позиция курсора"]
        SporeAnalyzer["`🔍 SporeAnalyzer`<br/>Анализ связей"]
    end

    subgraph "📦 Существующие системы"
        SharedDeps["`🔧 SharedDependencies`<br/>Общие вычисления"]
        ZoomManager["`🔍 ZoomManager`<br/>Трансформации"]
        SporeManager["`🧬 SporeManager`<br/>Все споры"`]
        InputManager["`🎹 InputManager`<br/>Обработка ввода"]
    end

    subgraph "👻 Система призраков"
        PredictionManager["`👻 PredictionManager`<br/>Призрачные деревья"]
        PreviewManager["`👁️ PreviewManager`<br/>Превью споры"]
        GhostToggle["`🔄 GhostToggle`<br/>Включение/выключение"]
    end

    ManualPicker --> SporeIndex
    ManualPicker --> CursorDetector
    ManualPicker --> SporeAnalyzer
    
    CursorDetector --> SharedDeps
    SporeIndex --> SporeManager
    
    InputManager --> ManualPicker
    InputManager --> GhostToggle
    
    GhostToggle --> PredictionManager
    GhostToggle --> PreviewManager
```

## 📁 Структура проекта v16

```
spores/v16_picker/
├── 📄 README.md           # Этот файл - главная документация
├── 🎮 scripts/run/        # Точки входа
│   └── main_demo.py      # Демо с пикингом спор
├── ⚙️ config/json/        # Конфигурация
│   ├── config.json       # Основные настройки + picker
│   ├── colors.json       # Цветовые схемы
│   └── ui_offsets.json   # Позиции UI элементов
├── 📂 src/               # Исходный код
│   ├── 🧠 core/          # Интеграция logic+visual
│   │   └── spore.py      # Главный класс споры
│   ├── 🎮 managers/      # Оркестраторы
│   │   ├── spore_manager.py          # Управление всеми спорами
│   │   ├── manual_spore_manager.py   # Ручное создание спор
│   │   ├── input_manager.py          # Обработка всего ввода
│   │   ├── manual_picker.py          # 🆕 Пикинг спор под курсором
│   │   ├── dt_manager.py             # Управление временным шагом
│   │   └── zoom_manager.py           # Масштабирование и зум
│   ├── 🔧 logic/         # Чистая математика
│   │   ├── spatial_index.py          # 🆕 Пространственный индекс
│   │   └── spore_analyzer.py         # 🆕 Анализ спор и связей
│   ├── 🎨 visual/        # Визуализация и UI
│   │   ├── ui_setup.py   # Настройка интерфейса
│   │   └── picker_ui.py  # 🆕 UI для пикера
│   └── 🔧 utils/         # Утилиты
│       ├── watcher.py    # Автоперезапуск при изменениях
│       └── debug_output.py # Система отладочного вывода
└── 📚 docs/              # Документация
    └── picker_guide.md   # 🆕 Руководство по пикеру
```

## 🎮 Система управления v16

### **Создание и редактирование спор**
- **ЛКМ**: создать семью из 5 спор в позиции курсора
- **Ctrl+Z**: удалить последнюю группу спор
- **F**: создать новую спору от последней (эволюция)
- **G**: активировать случайную кандидатскую спору

### **🆕 Система пикинга спор**
- **Наведение курсора**: автоматическое определение спор под курсором
- **J**: включить/выключить призрачные предсказания
- **I**: детальная информация о выбранной споре
- **P**: анализ связей и потомков споры

### **Управление временем и масштабом**
- **Ctrl+Scroll**: изменить dt (временной шаг)
- **M**: сбросить dt к исходному
- **E/T**: зум камеры
- **R**: сброс зума
- **1/2**: масштаб спор

### **Информация и отладка**
- **I**: статистика истории групп
- **J**: переключение призрачных предсказаний (🆕)
- **H**: показать/скрыть весь UI
- **C**: полная очистка (требует Ctrl+Shift)

## 🔧 Система UI и обновлений

### **Колбэки через data_providers**
```python
data_providers = {
    'get_spore_count': lambda: len(spore_manager.objects),
    'get_picker_info': lambda: picker.get_current_selection(),  # 🆕
    'get_cursor_spores': lambda: picker.get_spores_under_cursor(),  # 🆕
    # ... другие поставщики данных
}
```

### **🆕 Picker UI элементы**
- Индикатор выбранной споры
- Список спор под курсором
- Информация о связях и потомках
- Статистика пространственного индекса

## 🧪 Конфигурация v16

### **config.json структура (дополнения)**
```json
{
  "picker": {
    "threshold": 0.01,
    "grid_cell_size": 0.02,
    "debug_terminal_output": true,
    "ui_enabled": true,
    "max_spores_per_selection": 5
  },
  "ghost_system": {
    "enabled": true,
    "toggle_key": "j",
    "preview_alpha": 0.5,
    "predictions_alpha": 0.4
  },
  "pendulum": {
    "damping": 0.3,
    "dt": 0.05,
    "max_control": 2.0
  },
  "spore": {
    "scale": 0.1,
    "goal_position": [3.14159, 0]
  }
}
```

## 🚀 Как работать с проектом v16

### **Быстрый старт**
1. Запустить: `python src/utils/watcher.py` (автоперезапуск)
2. Или напрямую: `python scripts/run/main_demo.py`
3. Редактировать код → сохранить → автоматический перезапуск

### **🆕 Исследование спор через пикинг**
1. Создать несколько спор ЛКМ в разных местах
2. Наводить курсор на споры - в терминале появится информация
3. Нажать **J** чтобы скрыть призраков для четкости
4. Использовать **P** для анализа выбранной споры

### **Добавление нового функционала v16**
1. **Picker logic** → добавить в `src/logic/`
2. **Picker visual** → добавить в `src/visual/`  
3. **Integration** → обновить `src/managers/manual_picker.py`
4. **UI** → добавить в `picker_ui.py` + `ui_setup.py`
5. **Config** → обновить `config.json` секцию `picker`

### **Отладка пикера**
- Включить `debug_terminal_output` в config.json
- Использовать J для переключения призраков
- Watcher показывает ошибки при перезапуске

## 🎯 Текущие возможности системы v16

✅ **Реализовано**:
- Интерактивное создание семей спор с призрачными предсказаниями
- Система удаления групп с защитой от случайных действий  
- Динамическое управление временным шагом
- Полноценный UI с автообновлением
- Конфигурируемые параметры через config.json
- Автоперезапуск для быстрой итерации

🚧 **В разработке (v16)**:
- ManualPicker для выбора спор под курсором
- SporeIndex для эффективного пространственного поиска
- Переключение видимости призрачной системы (J)
- Детальный анализ связей выбранных спор

## 📋 Важные моменты для работы с v16

1. **Всегда используй** `project_knowledge_search` для изучения кода
2. **Следуй архитектуре** - logic отдельно от visual
3. **Добавляй типизацию** и defensive programming  
4. **Используй эмоджи** в логах для читаемости
5. **Тестируй с watcher** - мгновенная обратная связь
6. **🆕 Учитывай spatial indexing** при работе со спорами
7. **🆕 Проверяй real_position** а не transformed при пикинге
8. **Обновляй UI через колбэки** - не напрямую

---

**Проект активно развивается в направлении интерактивного исследования! v16 добавляет мощные возможности анализа спор через систему пикинга.** 🎯
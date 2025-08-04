# 📂 Исходный код (src) — v13_manual

> 🎮 **Система ручного создания спор с интерактивным превью**

Эта директория содержит весь исходный код проекта "Споры" версии 13_manual. Архитектура построена на принципах чистого кода с четким разделением ответственности между слоями.

![Архитектура](https://img.shields.io/badge/архитектура-clean_architecture-green)
![Версия](https://img.shields.io/badge/версия-v13_manual-blue)
![Паттерны](https://img.shields.io/badge/паттерны-manager_pattern-orange)

## 🏗️ Архитектурная схема v13_manual

```mermaid
graph TB
    subgraph "🖱️ v13_manual: Manual Input System"
        ManualSporeManager["`🎮 ManualSporeManager`<br/>Ручное создание спор"]
        CursorPreview["`👻 Cursor Preview`<br/>Полупрозрачная спора"]
        MouseHandler["`🖱️ Mouse Handler`<br/>Обработка ЛКМ"]
        PredictionChain["`🔮 Prediction Chain`<br/>Min/Max управление"]
    end

    subgraph "👀 Development Tools"
        Watcher["`⚡ Watcher`<br/>Автоперезапуск"]
        DebugOutput["`🐛 Debug Output`<br/>Система отладки"]
    end

    subgraph "⚡ Enhanced Managers"
        InputManager["`🎹 InputManager`<br/>+ ЛКМ обработка"]
        UpdateManager["`🔄 UpdateManager`<br/>+ Cursor updates"]
        SporeManager["`🧬 SporeManager`<br/>+ Manual mode"]
    end

    subgraph "🧠 Core Integration"
        Spore["`🌟 Spore`<br/>Logic + Visual"]
        SporeLogic["`📐 SporeLogic`<br/>Чистая математика"]
        SporeVisual["`🎨 SporeVisual`<br/>3D представление"]
    end

    subgraph "🔬 Pure Logic"
        PendulumSystem["`⚖️ PendulumSystem`<br/>Физическая модель"]
        Optimizer["`🎯 SporeOptimizer`<br/>L-BFGS-B + DE"]
        PoissonDisk["`🎲 Poisson Disk`<br/>Равномерная генерация"]
    end

    Watcher --> ManualSporeManager
    Watcher --> DebugOutput
    
    ManualSporeManager --> CursorPreview
    ManualSporeManager --> PredictionChain
    ManualSporeManager --> SporeManager
    
    InputManager --> ManualSporeManager
    UpdateManager --> ManualSporeManager
    
    SporeManager --> Spore
    Spore --> SporeLogic
    Spore --> SporeVisual
    
    SporeLogic --> PendulumSystem
    SporeManager --> Optimizer
```

## 📁 Структура директорий

### 🧠 **`core/`** — Интеграционное ядро
> Объединяет логику и визуализацию в единые сущности

```python
core/spore.py           # 🌟 Spore = SporeLogic + SporeVisual
core/simulation_engine.py  # 🎮 Альтернативный движок (не используется в main_demo)
```

**Ключевые концепции:**
- **Агрегация логики и визуализации** в одном объекте
- **Синхронизация** между 2D математикой и 3D отображением
- **Обратная совместимость** с существующим API

### 🔬 **`logic/`** — Чистая математика
> Независимые от Ursina алгоритмы и вычисления

```python
logic/pendulum.py           # ⚖️ Система маятника + кэширование матриц
logic/spore_logic.py        # 🧬 Логика споры (состояние, эволюция, стоимость)
logic/cost_function.py      # 💰 Расчет стоимости и градиента  
logic/spawn_area.py         # 🎯 Эллиптическая область + диски Пуассона
logic/ghost_processor.py    # 👻 Эффективный расчет предсказаний
logic/optimizer.py          # 🎯 L-BFGS-B и Differential Evolution
```

**Математические основы:**
- **Линеаризация и дискретизация** нелинейной системы
- **Оптимизация управления** с ограничениями
- **Пуассоновы диски** для равномерного размещения
- **Кэширование вычислений** для производительности

### 🎨 **`visual/`** — Отображение и UI
> Все что связано с Ursina Engine и 3D визуализацией

```python
visual/spore_visual.py          # 🧬 3D представление споры
visual/cost_visualizer.py       # 🗻 3D поверхность стоимости
visual/spawn_area_visualizer.py # 🎯 Визуализация области спавна  
visual/prediction_visualizer.py # 🔮 Призраки предсказаний
visual/scene_setup.py           # 🏗️ Настройка 3D сцены
visual/ui_manager.py            # 📱 Универсальная система UI
visual/ui_setup.py              # 📋 Готовые интерфейсы для демо
```

**Принципы визуализации:**
- **Модульность** — каждый компонент независим
- **Производительность** — буферизация и кэширование
- **Гибкость** — легко менять стили и поведение

### ⚡ **`managers/`** — Системные менеджеры
> Оркестраторы глобальных подсистем

```python
# 🆕 v13_manual: Новый ключевой компонент
managers/manual_spore_manager.py  # 🖱️ Ручное создание спор + превью

# Существующие менеджеры (обновлены для v13_manual)
managers/input_manager.py         # 🎹 Обработка ввода + ЛКМ
managers/update_manager.py        # 🔄 Цикл обновления + cursor updates
managers/spore_manager.py         # 🧬 Управление коллекцией спор
managers/zoom_manager.py          # 🔍 Масштабирование + look_point
managers/angel_manager.py         # 👼 Визуализация стоимости + переключение
managers/color_manager.py         # 🎨 Управление цветами из конфига
managers/param_manager.py         # ⚙️ Глобальный изменяемый параметр
managers/window_manager.py        # 🖥️ Настройка окна для разных мониторов
managers/spawn_area_manager.py    # 🎯 Управление областью спавна (отключен в v13)
```

**Паттерн Manager:**
- **Единая ответственность** за каждую подсистему
- **Слабое зацепление** между компонентами
- **Легкость тестирования** и замены реализаций

### 🛠️ **`utils/`** — Утилиты и вспомогательные функции
```python
utils/scalable.py           # 📏 Базовый класс для масштабируемых объектов
utils/poisson_disk.py       # 🎲 Алгоритм дисков Пуассона (Bridson)
utils/debug_output.py       # 🐛 Централизованная система отладки
utils/watcher.py            # 👀 Автоперезапуск при изменении файлов (🆕 v13_manual)
utils/ursina_patcher.py     # 🔧 Патчи для Ursina (множественные папки ассетов)
```

**Система отладки v13_manual:**
```python
from utils.debug_output import DebugOutput

debug = DebugOutput(config)
debug.print_verbose("Общая информация")           # Включается debug.enable_verbose_output
debug.print_evolution("Детали эволюции")         # Включается debug.enable_detailed_evolution  
debug.print_candidate("Информация о кандидатах")  # Включается debug.enable_candidate_logging
debug.print_trajectory("Траектории")             # Включается debug.enable_trajectory_logging
```

### ⚙️ **`config/`** — Конфигурация системы
```python
config/paths.py             # 📂 Управление путями к файлам проекта
config/ursina_setup.py      # 🎮 Глобальные настройки Ursina
```

## 🌊 Поток данных v13_manual

### 1. **🚀 Инициализация (main_demo.py)**
```python
# 1. Создание базовых систем
pendulum = PendulumSystem(damping=0.1)
zoom_manager = ZoomManager(scene_setup)
spore_manager = SporeManager(pendulum, zoom_manager, ...)

# 2. 🆕 v13_manual: Система ручного создания
manual_spore_manager = ManualSporeManager(
    spore_manager=spore_manager,
    zoom_manager=zoom_manager,
    pendulum=pendulum,
    color_manager=color_manager,
    config=config
)

# 3. Интеграция с менеджерами ввода/обновления
input_manager = InputManager(..., manual_spore_manager=manual_spore_manager)
update_manager = UpdateManager(..., manual_spore_manager=manual_spore_manager)
```

### 2. **🔄 Цикл обновления (каждый кадр)**
```python
def update():
    # 1. Обработка непрерывного ввода (включая ЛКМ)
    input_manager.update()
    
    # 2. 🆕 v13_manual: Обновление позиции курсора превью
    look_point = zoom_manager.identify_invariant_point()
    manual_spore_manager.update_cursor_position()
    
    # 3. Обновление всех UI элементов
    ui_setup.update()
```

### 3. **🎹 Обработка пользовательского ввода**
```python
def input(key):
    if key in ['q', 'escape']:
        application.quit()
    
    # Делегирование в централизованный InputManager
    input_manager.handle_input(key)
    
    # 🆕 ЛКМ обрабатывается в input_manager.update() через mouse.left
```

### 4. **🖱️ Ручное создание спор (v13_manual)**
```python
# В ManualSporeManager:
def update_cursor_position(self):
    # 1. Получаем позицию курсора через ZoomManager
    look_point = self.zoom_manager.identify_invariant_point()
    
    # 2. Обновляем превью спору
    self.preview_spore.position = look_point
    
    # 3. Обновляем предсказания min/max управления
    self.update_predictions(look_point)

def on_left_click(self):
    # Создаем настоящую спору + 2 дочерние по предсказаниям
    parent_spore = self.create_spore_at_cursor()
    min_child = self.create_prediction_spore(parent_spore, control_type='min')
    max_child = self.create_prediction_spore(parent_spore, control_type='max')
```

## 🎯 Принципы архитектуры

### **Разделение ответственности:**
- **Logic** — чистая математика, независимая от UI
- **Visual** — только отображение, не содержит бизнес-логики
- **Core** — интеграционный слой между logic и visual
- **Managers** — оркестрация и координация подсистем

### **Производительность (v12-v13_manual):**
- ✅ **Кэширование матриц** в PendulumSystem (expm, linearization)
- ✅ **Буферы массивов** в PredictionVisualizer  
- ✅ **Эффективные конвертации** 2D↔3D
- ✅ **Переиспользование объектов** в v13_manual

### **Тестируемость:**
- **Модульность** — легко покрыть тестами отдельные части
- **Инъекция зависимостей** — можно мокать компоненты
- **Чистые функции** — предсказуемое поведение

## 🔧 Руководство по разработке

### **Где что искать:**
- **Нужна математика?** → `logic/`
- **Проблема с отображением?** → `visual/`  
- **Интеграция компонентов?** → `core/`
- **Управление системой?** → `managers/`
- **Вспомогательная функция?** → `utils/`

### **Добавление новой функции:**
1. **Логика** в `logic/` (если нужны вычисления)
2. **Визуализация** в `visual/` (если нужно отображение)
3. **Интеграция** в `core/` (если связываете логику+визуал)
4. **Управление** в `managers/` (если нужен оркестратор)
5. **Настройка** в `main_demo.py`

### **Отладка проблем:**
```python
# Включите отладку для нужной области
# В config/json/config.json:
{
  "debug": {
    "enable_detailed_evolution": true,  # Для проблем с эволюцией
    "enable_candidate_logging": true,   # Для проблем с кандидатами
    "enable_trajectory_logging": true   # Для проблем с оптимизацией
  }
}
```

### **Использование watcher для разработки:**
```bash
# Запустите один раз
python src/utils/watcher.py

# Редактируйте любой .py файл
# Сохраните — автоматический перезапуск
# Наслаждайтесь мгновенной обратной связью!
```

## 📈 Метрики качества кода

### **Производительность:**
- ✅ **Кэширование матриц** в PendulumSystem
- ✅ **Буферы массивов** в PredictionVisualizer  
- ✅ **Эффективные конвертации** 2D↔3D
- ✅ **Переиспользование объектов** в v13_manual

### **Читаемость:**
- ✅ **Понятные имена** классов и методов
- ✅ **Единый стиль** кодирования
- ✅ **Хорошая структура** файлов и папок
- ✅ **Документированные интерфейсы**

### **Поддерживаемость:**
- ✅ **Модульная архитектура** — легко менять части
- ✅ **Конфигурируемость** через JSON файлы
- ✅ **Система отладки** для диагностики проблем
- ✅ **Обратная совместимость** при изменениях

---

**🎯 Начните изучение с `core/spore.py` — это сердце системы, которое объединяет всю логику и визуализацию!**
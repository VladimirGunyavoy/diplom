# UI Manager - Централизованное управление интерфейсом

## Обзор

**UI Manager** - это централизованная система управления всеми текстовыми элементами интерфейса в проекте spores. Она заменяет разрозненное создание `Text` элементов в различных классах единой системой управления.

## Проблемы, которые решает

### До UI Manager:
- ❌ Text элементы создавались в каждом классе отдельно
- ❌ Разный стиль оформления
- ❌ Дублирование кода для создания UI
- ❌ Сложность в управлении видимостью
- ❌ Отсутствие централизованного обновления

### После UI Manager:
- ✅ Все UI элементы в одном месте
- ✅ Единый стиль и цветовая схема
- ✅ Автоматическое обновление динамических элементов
- ✅ Группировка по категориям
- ✅ Простое API для управления видимостью

## Архитектура

### Основные компоненты

```python
UIManager
├── elements: Dict[str, Dict[str, Text]]  # Элементы по категориям
├── update_functions: Dict[str, Callable] # Функции обновления
├── styles: Dict[str, dict]              # Предустановленные стили
└── color_manager: ColorManager          # Управление цветами
```

### Категории элементов

1. **position** - Позиционная информация (координаты, поворот)
2. **instructions** - Инструкции управления
3. **status** - Статусные сообщения (блокировка курсора и т.д.)
4. **counters** - Счетчики и метрики
5. **info** - Информационные блоки
6. **debug** - Отладочная информация

## API

### Основные методы

#### Создание элементов

```python
# Общий метод создания
ui_manager.create_element(
    category='position',
    name='main',
    text="Position: 0, 0, 0",
    position=(0.4, -0.4),
    style='status'
)

# Специализированные методы
ui_manager.create_position_info('main')
ui_manager.create_instructions('controls', "WASD - движение")
ui_manager.create_counter('spores', 0, prefix="Спор: ")
ui_manager.create_status_indicator('cursor', "Cursor locked")
ui_manager.create_info_block('stats', "СТАТИСТИКА", "Загрузка...")
ui_manager.create_debug_info('performance')
```

#### Обновление элементов

```python
# Обновление текста
ui_manager.update_text('position', 'main', "Position: 1, 2, 3")

# Обновление счетчика
ui_manager.update_counter('spores', 42, prefix="Спор: ")

# Обновление информационного блока
ui_manager.update_info_block('stats', "СТАТИСТИКА", "Элементов: 10")
```

#### Управление видимостью

```python
# Отдельные элементы
ui_manager.show_element('position', 'main')
ui_manager.hide_element('position', 'main')
ui_manager.toggle_element('position', 'main')

# Целые категории
ui_manager.show_category('instructions')
ui_manager.hide_category('debug')
```

#### Автоматическое обновление

```python
# Регистрация функции обновления
def update_position():
    pos = player.position
    ui_manager.update_text('position', 'main', f"Position: {pos}")

ui_manager.register_update_function('position_update', update_position)

# Обновление всех динамических элементов
ui_manager.update_dynamic_elements()
```

### Стили

UI Manager предоставляет предустановленные стили:

```python
styles = {
    'default': {
        'scale': 0.75,
        'color': color_manager.get_color('ui', 'text_primary'),
        'background': True,
        'background_color': color_manager.get_color('ui', 'background_transparent'),
    },
    'header': {
        'scale': 0.9,
        'color': color_manager.get_color('ui', 'text_primary'),
        'background': True,
        'background_color': color_manager.get_color('ui', 'background_solid'),
    },
    'status': {
        'scale': 0.7,
        'color': color_manager.get_color('ui', 'text_secondary'),
        'background': False,
    },
    'counter': {
        'scale': 0.8,
        'color': color.yellow,
        'background': False,
    },
    'debug': {
        'scale': 0.6,
        'color': color.cyan,
        'background': True,
        'background_color': color_manager.get_color('ui', 'background_transparent'),
    }
}
```

## Интеграция с существующими классами

### SceneSetup

```python
# До
self.position_info = Text(text="", position=(0.4, -0.4), ...)
self.cursor_status = Text(text="Cursor locked", position=(0, -0.45), ...)

# После
self.ui_elements = StandardUISetup.setup_scene_ui(self.ui_manager, self)
```

### ZoomManager

```python
# До
self.instructions = Text(text="ZOOM: E - zoom in...", ...)
self.look_point_text = Text(text="LOOK POINT:", ...)

# После
self.ui_elements = StandardUISetup.setup_zoom_ui(self.ui_manager, self)
```

### ParamManager

```python
# До
self.instruction_text = Text(text="", position=(0, 0), ...)

# После
self.param_text = self.ui_manager.create_element(
    'counters', 'param_value',
    text=f'param value: {self.param}',
    style='header'
)
```

## StandardUISetup

Класс `StandardUISetup` предоставляет готовые настройки UI для типичных сценариев:

### setup_scene_ui()
- Позиционная информация камеры
- Статус блокировки курсора
- Инструкции управления сценой
- Автоматическое обновление позиции

### setup_zoom_ui()
- Инструкции управления зумом
- Информация о точке взгляда
- Автоматическое обновление look point

### setup_architecture_demo_ui()
- Информация об архитектуре
- Счетчик операций
- Инструкции тестирования

## Пример использования

```python
# Создание UI Manager
color_manager = ColorManager()
ui_manager = UIManager(color_manager)

# Создание сцены с UI Manager
scene_setup = SceneSetup(
    color_manager=color_manager,
    ui_manager=ui_manager
)

# Создание дополнительных элементов
spore_counter = ui_manager.create_counter(
    'spore_count', 0, 
    position=(-0.95, -0.1), 
    prefix="Спор: "
)

# Регистрация функции обновления
def update_spore_count():
    count = len(spore_manager.objects)
    ui_manager.update_counter('spore_count', count, prefix="Спор: ")

ui_manager.register_update_function('spore_count_update', update_spore_count)

# В главном цикле
def update():
    # Обновляем все UI элементы
    ui_manager.update_dynamic_elements()
```

## Миграция существующего кода

### Шаг 1: Добавить UI Manager в класс
```python
def __init__(self, ..., ui_manager=None):
    if ui_manager is None:
        ui_manager = UIManager(color_manager)
    self.ui_manager = ui_manager
```

### Шаг 2: Заменить создание Text элементов
```python
# Вместо
self.my_text = Text(text="Hello", position=(0, 0), ...)

# Используйте
self.my_text = self.ui_manager.create_element(
    'info', 'my_text', text="Hello", position=(0, 0)
)
```

### Шаг 3: Заменить обновление текста
```python
# Вместо
self.my_text.text = "New text"

# Используйте
self.ui_manager.update_text('info', 'my_text', "New text")
```

### Шаг 4: Добавить автообновление
```python
def my_update_function():
    self.ui_manager.update_text('info', 'my_text', get_dynamic_text())

self.ui_manager.register_update_function('my_update', my_update_function)
```

## Демонстрация

Запустите `scripts/3.py` для демонстрации UI Manager:

```bash
cd spores/7
python scripts/3.py
```

### Команды демонстрации:
- **U** - показать статистику UI в консоли
- **Y** - переключить отладочную информацию
- **I** - переключить инструкции
- **O** - переключить счетчики
- **P** - скрыть/показать все UI
- **M** - показать категории элементов

## Преимущества

### 1. Централизация
Все UI элементы управляются из одного места, что упрощает поддержку и модификацию.

### 2. Консистентность
Единый стиль оформления и цветовая схема через ColorManager.

### 3. Производительность
Автоматическое обновление только изменившихся элементов.

### 4. Гибкость
Легкое управление видимостью групп элементов и отдельных компонентов.

### 5. Расширяемость
Простое добавление новых типов UI элементов и стилей.

## Статистика интеграции

### Найденные UI элементы в проекте:

| Файл | Элементы | Статус |
|------|----------|--------|
| `scene_setup.py` | position_info, cursor_status, instructions | ✅ Мигрировано |
| `zoom_manager.py` | txt, instructions, look_point_text | ✅ Мигрировано |
| `param_manager.py` | instruction_text | ✅ Мигрировано |
| `scripts/2.py` | architecture_info, operation_info, test_info | ✅ Совместимо |

**Всего**: ~10 Text элементов централизованы в UI Manager.

## Будущие улучшения

1. **Анимации**: Добавить поддержку анимаций появления/исчезновения
2. **Темы**: Система тем для быстрой смены стилей
3. **Локализация**: Поддержка многоязычности
4. **Интерактивность**: Добавить кликабельные элементы
5. **Макеты**: Система автоматического позиционирования элементов 
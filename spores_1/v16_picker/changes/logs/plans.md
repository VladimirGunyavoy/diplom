# План создания автогенерируемого окна управления

## 📋 Задача
Создать автоматически генерируемое окно с инструкциями управления на основе зарегистрированных команд в InputManager.

## 🎯 Текущая ситуация

### Что уже есть:
1. **InputManager** с командной системой (`self.commands`)
2. Каждая команда содержит:
   - `description` - описание команды
   - `category` - категория (споры, зум, визуализация и т.д.)
   - `handler` - обработчик
   - `enabled` - функция проверки доступности
3. Метод `get_commands_by_category()` - группировка по категориям
4. UI_setup для создания UI элементов

### Чего не хватает:
- UI элемента для отображения команд
- Автоматической генерации текста из `self.commands`
- Интеграции с UI_setup

## 📝 Пошаговый план реализации

### Артефакт 1: Создание ControlsWindow класса
**Файл**: `src/visual/controls_window.py`

```python
from ursina import Text, Entity
from typing import Dict, List, Tuple, Optional

class ControlsWindow:
    """Окно с автогенерируемыми инструкциями управления."""
    
    def __init__(self, input_manager, color_manager=None):
        self.input_manager = input_manager
        self.color_manager = color_manager
        self.window_entity = None
        self.text_elements = []
        self.visible = True
        
        self._create_window()
        self._generate_controls_text()
    
    def _create_window(self):
        """Создает базовое окно."""
        # Создать Entity для фона окна
        # Позиция: левый верхний угол
        # Добавить полупрозрачный черный фон
        
    def _generate_controls_text(self):
        """Генерирует текст из команд InputManager."""
        # Получить команды через input_manager.get_commands_by_category()
        # Отфильтровать только включенные команды
        # Сгруппировать по категориям
        # Создать Text элементы для каждой категории
        
    def _format_key_description(self, key: str, description: str) -> str:
        """Форматирует строку команды."""
        # Привести key к читаемому виду (scroll up → Scroll↑)
        # Форматировать: "W/A/S/D - move camera"
        
    def toggle_visibility(self):
        """Переключает видимость окна."""
        
    def update_commands(self):
        """Обновляет список команд (если изменились)."""
```

### Артефакт 2: Интеграция с InputManager
**Изменения в**: `src/managers/input_manager.py`

```python
# В __init__ добавить:
self.controls_window = None

# Новый метод:
def set_controls_window(self, controls_window):
    """Связывает окно управления с InputManager."""
    self.controls_window = controls_window

# В _setup_command_system добавить команду:
'?': {
    'description': 'показать/скрыть окно управления',
    'handler': self._handle_toggle_controls,
    'category': 'интерфейс',
    'enabled': lambda: True
}

def _handle_toggle_controls(self):
    """Переключает видимость окна управления."""
    if self.controls_window:
        self.controls_window.toggle_visibility()
```

### Артефакт 3: Добавление в UI_setup
**Изменения в**: `src/visual/ui_setup.py`

```python
# В setup_demo_ui добавить:
def setup_demo_ui(self, data_providers, spawn_area=None, input_manager=None):
    # ... существующий код ...
    
    # Создание окна управления
    if input_manager:
        from ..visual.controls_window import ControlsWindow
        controls_window = ControlsWindow(
            input_manager=input_manager,
            color_manager=self.color_manager
        )
        input_manager.set_controls_window(controls_window)
        self.controls_window = controls_window
```

### Артефакт 4: Добавление в main_demo.py
**Изменения в**: `scripts/run/main_demo.py`

```python
# При вызове setup_demo_ui передать input_manager:
ui_elements = ui_setup.setup_demo_ui(
    data_providers, 
    spawn_area=spawn_area_logic,
    input_manager=input_manager  # Добавить эту строку
)
```

## 🎨 Детали реализации

### Форматирование клавиш:
```python
KEY_DISPLAY_NAMES = {
    'scroll up': 'Scroll↑',
    'scroll down': 'Scroll↓',
    'left mouse down': 'ЛКМ',
    'right mouse down': 'ПКМ',
    'space': 'Space',
    'shift': 'Shift',
    'escape': 'Esc',
    'ctrl+z': 'Ctrl+Z',
    # ... остальные
}
```

### Группировка категорий:
```python
CATEGORY_DISPLAY = {
    'камера': '📷 КАМЕРА',
    'споры': '🌱 СПОРЫ',
    'зум': '🔍 ЗУМ',
    'визуализация': '👁️ ВИЗУАЛИЗАЦИЯ',
    'время': '⏰ ВРЕМЯ',
    # ... остальные
}
```

### Компактное отображение:
```python
# Вместо отдельных строк для каждой команды:
"W/A/S/D - движение камеры"
"Space/Shift - вверх/вниз"
"E/T - зум in/out"
"R - сброс зума"
```

## 🔄 Дополнительные улучшения

### 1. Динамическое обновление
- При изменении режима (spores → tree) обновлять отображаемые команды
- Показывать только доступные команды (enabled == True)

### 2. Контекстная фильтрация
```python
def filter_by_context(self, context='all'):
    """Фильтрует команды по контексту."""
    # all - все команды
    # basic - только основные
    # advanced - расширенные команды
```

### 3. Цветовое кодирование
- ✅ Зеленым - доступные команды
- 🔶 Оранжевым - контекстные команды (зависят от режима)
- ❌ Серым - недоступные команды

### 4. Позиционирование
```python
# Варианты размещения:
POSITIONS = {
    'top_left': (-0.8, 0.45),
    'top_right': (0.5, 0.45),
    'bottom_left': (-0.8, -0.45),
    'bottom_right': (0.5, -0.45),
}
```

## 📊 Ожидаемый результат

### Окно будет показывать:
```
╔═══════════════════════╗
║     CONTROLS:         ║
║ WASD - move camera    ║
║ Space/Shift - up/down ║
║ E/T - zoom in/out     ║
║ R - reset zoom        ║
║ 1/2 - scale spores    ║
║ ───────────────────   ║
║ F - evolve spore      ║
║ G - activate random   ║
║ V - evolve all        ║
║ ───────────────────   ║
║ H - hide/show UI      ║
║ M - reset dt          ║
║ ? - show this help    ║
╚═══════════════════════╝
```

## 🚀 Последовательность действий для агента

1. **Создай файл** `src/visual/controls_window.py` с классом ControlsWindow
2. **Добавь импорт** в `src/managers/input_manager.py`
3. **Добавь метод** `set_controls_window` в InputManager
4. **Добавь команду** `?` для переключения окна
5. **Интегрируй** в UI_setup
6. **Обнови вызов** в main_demo.py
7. **Протестируй** - окно должно появляться/скрываться по `?`

## ⚠️ Важные моменты

1. **Не дублировать** существующие UI элементы
2. **Использовать** существующие стили из ColorManager
3. **Учитывать** разные мониторы (через UI_POSITIONS)
4. **Кешировать** сгенерированный текст (обновлять только при изменениях)
5. **Проверять** enabled() для каждой команды

## 🎯 Критерии успеха

✅ Окно автоматически генерируется из `self.commands`  
✅ Показывает только доступные команды  
✅ Группирует по категориям  
✅ Переключается по клавише `?`  
✅ Обновляется при изменении режима  
✅ Использует единый стиль с остальным UI

"""
Controls Window - Автогенерируемое окно с инструкциями управления
================================================================

Создает окно с командами управления, автоматически генерируя
содержимое из зарегистрированных команд InputManager.
"""

from ursina import Text, Entity, color
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..managers.input_manager import InputManager
    from ..managers.color_manager import ColorManager


class ControlsWindow:
    """
    Окно с автогенерируемыми инструкциями управления.
    
    Автоматически создает список команд из InputManager
    и отображает их в удобном виде с группировкой по категориям.
    """
    
    # Маппинг клавиш для красивого отображения
    KEY_DISPLAY_NAMES = {
        'scroll up': 'Scroll↑',
        'scroll down': 'Scroll↓',
        'wheel up': 'Scroll↑',
        'wheel down': 'Scroll↓',
        'left mouse down': 'ЛКМ',
        'right mouse down': 'ПКМ',
        'space': 'Space',
        'shift': 'Shift',
        'escape': 'Esc',
        'ctrl+z': 'Ctrl+Z',
        'ctrl+c': 'Ctrl+C',
        'ctrl+scroll': 'Ctrl+Scroll',
        'ctrl+shift+c': 'Ctrl+Shift+C',
        'alt': 'Alt',
        ':': ':',
    }
    
    # Порядок и названия категорий для отображения
    CATEGORY_ORDER = [
        ('камера', 'CAMERA'),
        ('споры', 'SPORES'),
        ('зум', 'ZOOM'),
        ('время', 'TIME'),
        ('визуализация', 'VISUAL'),
        ('дерево', 'TREE'),
        ('отладка', 'DEBUG'),
        ('интерфейс', 'UI'),
    ]
    
    def __init__(self, 
                 input_manager: 'InputManager',
                 color_manager: Optional['ColorManager'] = None,
                 position: Tuple[float, float] = (-0.88, 0.45),
                 scale: float = 0.7):
        """
        Инициализация окна управления.
        
        Args:
            input_manager: Менеджер команд для получения списка
            color_manager: Менеджер цветов для стилизации
            position: Позиция окна на экране (нормализованные координаты)
            scale: Масштаб текста
        """
        self.input_manager = input_manager
        self.color_manager = color_manager
        self.position = position
        self.scale = scale
        self.visible = True
        
        # UI элементы
        self.background = None
        self.title_text = None
        self.controls_text = None
        
        # Кеш для сгенерированного текста
        self._cached_text = ""
        self._last_commands_hash = None
        
        # Создаем окно
        self._create_window()
        self._update_controls_text()
    
    def _create_window(self):
        """Создает визуальные элементы окна."""
        # Фон окна
        self.background = Entity(
            parent=camera.ui,
            model='quad',
            color=color.rgba(20, 20, 30, 220),  # Полупрозрачный темный фон
            position=(self.position[0], self.position[1], -1),
            scale=(0.35, 0.5, 1),
            enabled=self.visible
        )
        
        # Заголовок
        self.title_text = Text(
            'CONTROLS:',
            parent=camera.ui,
            position=(self.position[0] - 0.15, self.position[1] + 0.23),
            scale=self.scale * 1.2,
            color=color.white if not self.color_manager else self.color_manager.get_color('ui', 'title'),
            enabled=self.visible
        )
        
        # Текст с командами
        self.controls_text = Text(
            '',
            parent=camera.ui,
            position=(self.position[0] - 0.15, self.position[1] + 0.18),
            scale=self.scale,
            color=color.light_gray if not self.color_manager else self.color_manager.get_color('ui', 'text'),
            enabled=self.visible
        )
    
    def _format_key(self, key: str) -> str:
        """
        Форматирует клавишу для отображения.
        
        Args:
            key: Название клавиши из InputManager
            
        Returns:
            Отформатированное название
        """
        # Используем маппинг если есть
        if key in self.KEY_DISPLAY_NAMES:
            return self.KEY_DISPLAY_NAMES[key]
        
        # Для букв и цифр - просто uppercase
        if len(key) == 1:
            return key.upper()
        
        # Для остального - capitalize
        return key.replace('_', ' ').title()
    
    def _group_similar_commands(self, commands: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Группирует похожие команды для компактного отображения.
        
        Например: W, A, S, D -> WASD - move camera
        """
        # Группируем команды движения
        movement_keys = []
        movement_desc = ""
        zoom_keys = []
        zoom_desc = ""
        scale_keys = []
        scale_desc = ""
        
        grouped = []
        skip_keys = set()
        
        for key, desc in commands:
            if key in skip_keys:
                continue
                
            # Группировка движения камеры
            if key in ['w', 'a', 's', 'd'] and 'камер' in desc.lower():
                movement_keys.append(self._format_key(key))
                movement_desc = "move camera"
                skip_keys.update(['w', 'a', 's', 'd'])
            
            # Группировка вертикального движения
            elif key in ['space', 'shift'] and ('вверх' in desc.lower() or 'вниз' in desc.lower()):
                if not movement_keys:  # Если еще не добавили WASD
                    grouped.append(("Space/Shift", "up/down"))
                skip_keys.update(['space', 'shift'])
            
            # Группировка зума
            elif key in ['e', 't'] and 'зум' in desc.lower():
                zoom_keys.append(self._format_key(key))
                zoom_desc = "zoom in/out"
                skip_keys.update(['e', 't'])
            
            # Группировка масштаба спор
            elif key in ['1', '2'] and 'спор' in desc.lower() and 'масштаб' in desc.lower():
                scale_keys.append(self._format_key(key))
                scale_desc = "scale spores"
                skip_keys.update(['1', '2'])
            
            # Остальные команды добавляем как есть
            elif key not in skip_keys:
                grouped.append((self._format_key(key), desc))
        
        # Добавляем сгруппированные команды
        if movement_keys:
            grouped.insert(0, ("WASD", movement_desc))
            grouped.insert(1, ("Space/Shift", "up/down"))
        if zoom_keys:
            grouped.insert(2 if movement_keys else 0, ("E/T", zoom_desc))
        if scale_keys:
            idx = 3 if movement_keys else (1 if zoom_keys else 0)
            grouped.insert(idx, ("1/2", scale_desc))
        
        return grouped
    
    def _generate_controls_text(self) -> str:
        """
        Генерирует текст с командами из InputManager.
        
        Returns:
            Отформатированный текст для отображения
        """
        if not self.input_manager or not hasattr(self.input_manager, 'get_commands_by_category'):
            return "Commands not available"
        
        lines = []
        commands_by_category = self.input_manager.get_commands_by_category()
        
        # Проходим по категориям в заданном порядке
        for cat_key, cat_display in self.CATEGORY_ORDER:
            if cat_key not in commands_by_category:
                continue
            
            commands = commands_by_category[cat_key]
            if not commands:
                continue
            
            # Фильтруем только включенные команды
            enabled_commands = []
            for key, desc in commands:
                if key in self.input_manager.commands:
                    cmd_info = self.input_manager.commands[key]
                    if cmd_info['enabled']():
                        # Укорачиваем описания для компактности
                        short_desc = desc.replace('переключить ', '')
                        short_desc = short_desc.replace('переключение ', '')
                        short_desc = short_desc.replace('показать/скрыть ', '')
                        enabled_commands.append((key, short_desc))
            
            if not enabled_commands:
                continue
            
            # Группируем похожие команды для категории "камера"
            if cat_key == 'камера':
                enabled_commands = self._group_similar_commands(enabled_commands)
            
            # Добавляем команды категории
            for key, desc in enabled_commands:
                formatted_key = self._format_key(key) if cat_key != 'камера' else key
                # Форматируем строку с выравниванием
                lines.append(f"{formatted_key:<12} - {desc}")
            
            # Разделитель между категориями (кроме последней)
            if cat_key != self.CATEGORY_ORDER[-1][0]:
                lines.append("")  # Пустая строка между категориями
        
        return '\n'.join(lines)
    
    def _update_controls_text(self):
        """Обновляет текст с командами."""
        new_text = self._generate_controls_text()
        
        # Обновляем только если текст изменился
        if new_text != self._cached_text:
            self._cached_text = new_text
            if self.controls_text:
                self.controls_text.text = new_text
    
    def toggle_visibility(self):
        """Переключает видимость окна."""
        self.visible = not self.visible
        
        if self.background:
            self.background.enabled = self.visible
        if self.title_text:
            self.title_text.enabled = self.visible
        if self.controls_text:
            self.controls_text.enabled = self.visible
        
        status = "показано" if self.visible else "скрыто"
        print(f"📋 Окно управления {status}")
    
    def update_commands(self):
        """Обновляет список команд (если изменились)."""
        self._update_controls_text()
    
    def set_position(self, position: Tuple[float, float]):
        """Изменяет позицию окна."""
        self.position = position
        
        if self.background:
            self.background.position = (position[0], position[1], -1)
        if self.title_text:
            self.title_text.position = (position[0] - 0.15, position[1] + 0.23)
        if self.controls_text:
            self.controls_text.position = (position[0] - 0.15, position[1] + 0.18)



# Добавить в начало файла src/managers/input_manager.py в блок импортов:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..visual.controls_window import ControlsWindow

# Добавить в __init__ после строки self.buffer_merge_manager = BufferMergeManager(...):
        # 🆕 v16: Окно с инструкциями управления
        self.controls_window: Optional['ControlsWindow'] = None

# Добавить новый метод после __init__:
    def set_controls_window(self, controls_window: 'ControlsWindow'):
        """
        Связывает окно управления с InputManager.
        
        Args:
            controls_window: Экземпляр ControlsWindow для отображения команд
        """
        self.controls_window = controls_window
        print("📋 Окно управления связано с InputManager")

# В методе _setup_command_system() добавить новую команду перед закрывающей скобкой словаря:
            # === СПРАВКА ===
            '?': {
                'description': 'показать/скрыть это окно справки',
                'handler': self._handle_toggle_controls,
                'category': 'интерфейс',
                'enabled': lambda: True
            },
            'slash': {  # Альтернативная клавиша для справки
                'description': 'показать/скрыть окно управления',
                'handler': self._handle_toggle_controls,
                'category': 'интерфейс', 
                'enabled': lambda: True
            },

# Добавить новый метод-обработчик после других _handle_* методов:
    def _handle_toggle_controls(self):
        """Обработчик переключения видимости окна управления."""
        if self.controls_window:
            self.controls_window.toggle_visibility()
        else:
            print("⚠️ Окно управления не инициализировано")

# В методе toggle_creation_mode (если есть) или handle_input добавить обновление окна:
        # Обновляем окно управления при смене режима
        if self.controls_window:
            self.controls_window.update_commands()



# ============================================
# Изменения в src/visual/ui_setup.py
# ============================================

# Добавить импорт в начало файла:
from typing import Optional

# Изменить сигнатуру метода setup_demo_ui, добавив параметр input_manager:
    def setup_demo_ui(self, data_providers, spawn_area=None, input_manager=None):
        """
        Настраивает полный UI для демо с колбэками для обновления.
        
        Args:
            data_providers: Словарь с функциями-поставщиками данных
            spawn_area: Опциональная область спавна для отображения статуса
            input_manager: Опциональный InputManager для создания окна управления
        """
        
        # ... существующий код ...
        
        # Добавить перед return ui_elements:
        
        # === ОКНО УПРАВЛЕНИЯ ===
        if input_manager:
            try:
                from ..visual.controls_window import ControlsWindow
                
                # Создаем окно управления
                self.controls_window = ControlsWindow(
                    input_manager=input_manager,
                    color_manager=self.color_manager,
                    position=(-0.88, 0.45),  # Левый верхний угол
                    scale=0.7
                )
                
                # Связываем с InputManager
                input_manager.set_controls_window(self.controls_window)
                
                print("   ✓ Окно управления создано и связано")
            except ImportError as e:
                print(f"   ⚠️ Не удалось создать окно управления: {e}")
                self.controls_window = None
        else:
            self.controls_window = None

# ============================================
# Изменения в scripts/run/main_demo.py
# ============================================

# Найти строку с вызовом ui_setup.setup_demo_ui (примерно строка 240-250):
# Изменить вызов, добавив параметр input_manager:

# БЫЛО:
ui_elements = ui_setup.setup_demo_ui(data_providers, spawn_area=spawn_area_logic)

# СТАЛО:
ui_elements = ui_setup.setup_demo_ui(
    data_providers, 
    spawn_area=spawn_area_logic,
    input_manager=input_manager  # Передаем InputManager для окна управления
)

# В секции вывода доступных команд (после "📋 ДОСТУПНЫЕ КОМАНДЫ:") добавить:
print("   СПРАВКА: ? или / (показать/скрыть окно управления)")

# Опционально: можно удалить или закомментировать старый список команд,
# так как теперь они будут в окне:
# print("   ОБЫЧНЫЕ: WASD, Space/Shift, мышь, Alt, Q")
# print("   СПОРЫ: F (эволюция), G (кандидат), V (развить всех)")
# ... и так далее
print("   💡 Нажмите ? для отображения полной справки по управлению")

# 🧪 Инструкции по тестированию и улучшениям

## 📝 Порядок реализации

1. **Создай файл** `src/visual/controls_window.py` (Артефакт 1)
2. **Внеси изменения** в `src/managers/input_manager.py` (Артефакт 2)
3. **Обнови** `src/visual/ui_setup.py` (Артефакт 3, первая часть)
4. **Измени вызов** в `scripts/run/main_demo.py` (Артефакт 3, вторая часть)
5. **Запусти** и протестируй

## ✅ Тестирование

### Базовая проверка:
```bash
python scripts/run/main_demo.py
```

1. **При запуске** - окно должно появиться в левом верхнем углу
2. **Нажми `?`** - окно должно скрыться
3. **Нажми `?` снова** - окно должно появиться
4. **Проверь команды** - все активные команды должны отображаться
5. **Переключи режим (K)** - окно должно обновиться (если добавишь update_commands)

### Проверка содержимого:
- ✅ Команды сгруппированы по категориям
- ✅ WASD объединены в одну строку
- ✅ Space/Shift объединены
- ✅ E/T для зума объединены
- ✅ Только доступные команды отображаются

## 🔧 Возможные проблемы и решения

### Проблема 1: Окно не появляется
```python
# Проверь в консоли:
print(f"Controls window created: {ui_setup.controls_window}")
print(f"InputManager has window: {input_manager.controls_window}")
```

### Проблема 2: Текст обрезается
```python
# В controls_window.py измени масштаб фона:
self.background = Entity(
    # ...
    scale=(0.4, 0.6, 1),  # Увеличить ширину и высоту
)
```

### Проблема 3: Клавиша ? не работает
```python
# Добавь отладку в input_manager.py:
def handle_input(self, key: str) -> None:
    print(f"[DEBUG] Received key: {key}")
    if key == '?':
        print("[DEBUG] Question mark detected!")
```

## 🎨 Дополнительные улучшения

### 1. Анимация появления/скрытия
```python
from ursina import Sequence, Func, Wait

def toggle_visibility(self):
    if not self.visible:
        # Анимация появления
        self.background.enabled = True
        self.background.animate_scale((0.35, 0.5, 1), duration=0.2, curve=curve.out_expo)
        self.background.animate('color', color.rgba(20, 20, 30, 220), duration=0.3)
    else:
        # Анимация исчезновения
        self.background.animate_scale((0, 0, 1), duration=0.2, curve=curve.in_expo)
        Sequence(Wait(0.2), Func(setattr, self.background, 'enabled', False)).start()
```

### 2. Подсветка изменений
```python
def highlight_changed_commands(self, changed_keys: list):
    """Подсвечивает измененные команды."""
    for key in changed_keys:
        # Временно изменить цвет строки с этой командой
        pass
```

### 3. Фильтрация по режимам
```python
def filter_by_mode(self, mode: str):
    """Показывает только команды для конкретного режима."""
    if mode == 'spores':
        # Скрыть команды дерева
        pass
    elif mode == 'tree':
        # Скрыть команды спор
        pass
```

### 4. Сохранение позиции окна
```python
def save_position_to_config(self):
    """Сохраняет позицию окна в конфиг."""
    config = {
        'controls_window': {
            'position': self.position,
            'visible': self.visible
        }
    }
    # Сохранить в config.json
```

### 5. Интерактивное перемещение
```python
def make_draggable(self):
    """Делает окно перетаскиваемым мышью."""
    self.background.draggable = True
    self.background.on_click = self.start_drag
```

### 6. Поиск по командам
```python
def add_search_field(self):
    """Добавляет поле поиска команд."""
    self.search_input = InputField(
        parent=camera.ui,
        position=(self.position[0], self.position[1] + 0.3)
    )
```

### 7. Экспорт в файл
```python
def export_to_file(self, filename='controls.txt'):
    """Экспортирует список команд в текстовый файл."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("УПРАВЛЕНИЕ СИМУЛЯЦИЕЙ\n")
        f.write("=" * 40 + "\n")
        f.write(self._cached_text)
    print(f"📄 Команды экспортированы в {filename}")
```

## 🎯 Проверка успешной реализации

После реализации должно работать следующее:

1. ✅ **Окно появляется** при запуске программы
2. ✅ **Клавиша `?`** переключает видимость
3. ✅ **Команды сгруппированы** по категориям
4. ✅ **Компактное отображение** (WASD, Space/Shift и т.д.)
5. ✅ **Только активные команды** отображаются
6. ✅ **Стиль соответствует** остальному UI
7. ✅ **Нет дублирования** с другими UI элементами

## 💡 Финальные советы

1. **Начни с базовой версии** - просто отображение команд
2. **Протестируй** основной функционал
3. **Добавляй улучшения** по одному
4. **Используй существующие стили** из ColorManager
5. **Следи за производительностью** - не обновляй текст каждый кадр
6. **Документируй изменения** в коде

## 🚀 Результат

После успешной реализации у тебя будет:
- Автогенерируемое окно с командами
- Всегда актуальный список управления
- Возможность скрыть/показать по клавише
- Группировка и форматирование команд
- Единый стиль с остальным интерфейсом

# ⚙️ Конфигурация — v13_manual

Эта директория и связанные с ней модули отвечают за всю конфигурацию проекта. Подход разделен на два типа:

1.  **JSON-конфигурация (`json/`)**: Для параметров, которые часто меняются и не должны требовать изменения кода (например, цвета, размеры, физические константы).
2.  **Python-конфигурация (`../src/config/`)**: Для системных настроек, связанных с кодом (настройка путей, инициализация движка).

![Версия](https://img.shields.io/badge/версия-v13_manual-blue)
![Конфигурация](https://img.shields.io/badge/config-JSON%2BPython-orange)

---

## 📄 JSON-конфигурация (`json/`)

Эти файлы позволяют гибко настраивать поведение и внешний вид симуляции без изменения кода.

### `config.json` — Основной конфигурационный файл

Содержит параметры для ключевых компонентов симуляции.

#### **🔬 Физическая система:**
```json
{
  "pendulum": {
    "damping": 0.1,           // Коэффициент затухания маятника
    "max_control": 1.0        // Максимальное управляющее воздействие
  }
}
```

#### **🧬 Параметры спор:**
```json
{
  "spore": {
    "initial_position": [0, 0],      // Начальная позиция [theta, theta_dot]
    "goal_position": [3.14159, 0],   // Целевая позиция (перевернутое положение)
    "scale": 0.05                    // Размер спор в 3D сцене
  }
}
```

#### **🎯 Область спавна (отключена в v13_manual):**
```json
{
  "spawn_area": {
    "eccentricity": 0.8,      // Эксцентриситет эллипса
    "min_radius": 0.3,        // Минимальный радиус для дисков Пуассона
    "enabled": false          // 🆕 v13_manual: отключена по умолчанию
  }
}
```

#### **🗻 Поверхность стоимости:**
```json
{
  "cost_surface": {
    "enabled": false,         // Включить/выключить отображение
    "alpha": 0.3,            // Прозрачность поверхности
    "show_edges": true,      // Показывать рёбра
    "show_contours": true    // Показывать контурные линии
  }
}
```

#### **👼 Визуализация функции стоимости ("ангелы"):**
```json
{
  "angel": {
    "show_angels": false,     // 🆕 v11-v13: управление видимостью
    "show_pillars": true,     // Показывать столбы под спорами
    "pillar_width": 0.02,     // Ширина столбов
    "pillar_alpha": 0.7       // Прозрачность столбов
  }
}
```

#### **🐛 Система отладки (v13_manual):**
```json
{
  "debug": {
    "enable_verbose_output": false,      // Общие отладочные сообщения
    "enable_detailed_evolution": false,  // Детальная информация об эволюции спор
    "enable_candidate_logging": false,   // Операции с кандидатскими спорами
    "enable_trajectory_logging": false   // Оптимизация траекторий и объединение спор
  }
}
```

#### **🔗 Визуальные связи между спорами:**
```json
{
  "link": {
    "show_links": true,       // Показывать связи между родителями и детьми
    "line_width": 2.0,        // Толщина линий связей
    "fade_alpha": 0.5         // Прозрачность неактивных связей
  }
}
```

#### **🎬 Начальные параметры сцены:**
```json
{
  "scene_setup": {
    "camera_position": [0, 5, -10],    // Начальная позиция камеры
    "camera_rotation": [15, 0, 0],     // Начальный поворот камеры
    "floor_scale": 20,                 // Размер пола
    "background_color": [0.1, 0.1, 0.1] // Цвет фона
  }
}
```

### `sizes.json` — Размеры элементов

Определяет размеры различных элементов UI и сцены для консистентности.

#### **🏗️ Элементы сцены:**
```json
{
  "scene": {
    "floor_scale": 20,        // Размер пола
    "axis_length": 5,         // Длина осей координат
    "grid_spacing": 1         // Расстояние между линиями сетки
  }
}
```

#### **📱 Элементы пользовательского интерфейса:**
```json
{
  "ui": {
    "text_size": 0.03,        // Размер обычного текста
    "header_size": 0.04,      // Размер заголовков
    "margin": 0.02,           // Отступы между элементами
    "panel_width": 0.3,       // Ширина панелей
    "button_height": 0.05     // Высота кнопок
  }
}
```

### `colors.json` — Цветовая палитра

Централизованное управление всеми цветами в приложении. `ColorManager` загружает цвета из этого файла.

#### **🏗️ Цвета сцены:**
```json
{
  "scene": {
    "floor": [0.2, 0.2, 0.2, 1],     // Серый пол
    "background": [0.1, 0.1, 0.1],   // Тёмно-серый фон
    "x_axis": [1, 0, 0, 1],          // Красная ось X
    "y_axis": [0, 1, 0, 1],          // Зелёная ось Y
    "z_axis": [0, 0, 1, 1]           // Синяя ось Z
  }
}
```

#### **🧬 Цвета спор:**
```json
{
  "spore": {
    "normal": [1, 1, 1, 1],          // Белая обычная спора
    "goal": [0, 1, 0, 1],            // Зелёная целевая спора
    "ghost": [1, 1, 1, 0.3],         // Полупрозрачный призрак
    "candidate": [1, 1, 1, 0.15],    // 🆕 v11: белый полупрозрачный кандидат
    "completed": [0, 0, 1, 1],       // 🆕 v11: синий цвет завершённых спор
    "preview": [1, 1, 0, 0.5]        // 🆕 v13_manual: жёлтый превью курсора
  }
}
```

#### **🔗 Цвета связей:**
```json
{
  "link": {
    "normal": [0.7, 0.7, 0.7, 0.8],  // Серые обычные связи
    "active": [1, 1, 0, 1],          // 🆕 v11: жёлтые активные связи
    "parent_child": [0, 1, 1, 0.6]   // Голубые связи родитель-ребёнок
  }
}
```

#### **🎯 Цвета области спавна:**
```json
{
  "spawn_area": {
    "ellipse": [0, 1, 1, 0.3],       // Голубой эллипс области
    "border": [0, 1, 1, 0.8]         // Более яркая граница
  }
}
```

#### **🗻 Цвета поверхности стоимости:**
```json
{
  "cost_surface": {
    "surface": [1, 0.5, 0, 0.3],     // Оранжевая поверхность
    "edges": [1, 0.5, 0, 0.6],       // Более яркие рёбра
    "contours": [1, 0, 0, 0.8]       // Красные контурные линии
  }
}
```

#### **👼 Цвета "ангелов":**
```json
{
  "angel": {
    "pillar": [1, 1, 1, 0.7],        // Белые столбы
    "high_cost": [1, 0, 0, 0.8],     // Красный для высокой стоимости
    "low_cost": [0, 1, 0, 0.8],      // Зелёный для низкой стоимости
    "medium_cost": [1, 1, 0, 0.8]    // Жёлтый для средней стоимости
  }
}
```

#### **📱 Цвета UI:**
```json
{
  "ui": {
    "text": [1, 1, 1, 1],            // Белый основной текст
    "header": [0, 1, 1, 1],         // Голубые заголовки
    "instructions": [0.7, 0.7, 0.7, 1], // Серые инструкции
    "warning": [1, 0.5, 0, 1],      // Оранжевые предупреждения
    "error": [1, 0, 0, 1]           // Красные ошибки
  }
}
```

---

## 🐍 Python-конфигурация (`../src/config/`)

Эти модули выполняют настройку на уровне кода при запуске приложения.

### `paths.py` — Управление путями

Централизованное управление путями к основным директориям проекта.

```python
import os
from pathlib import Path

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent
        self.config_dir = self.root / 'config'
        self.assets_dir = self.root / 'assets'
        self.logs_dir = self.root / 'log'
        self.src_dir = self.root / 'src'
        
    def get_config_path(self, filename):
        return self.config_dir / 'json' / filename
        
    def get_asset_path(self, filename):
        return self.assets_dir / filename
```

### `ursina_setup.py` — Настройки движка

Применяет глобальные настройки для движка `Ursina` перед инициализацией.

```python
from ursina import application

def setup_ursina_global_settings():
    """Настройки, которые должны быть применены до создания окна."""
    application.development_mode = False
    application.vsync = True
    application.show_fps_counter = True
    
    # Настройки окна
    application.window_fullscreen = False
    application.window_borderless = False
```

---

## 🔧 Использование конфигурации

### **Загрузка JSON конфигурации:**
```python
import json
from src.config.paths import ProjectPaths

paths = ProjectPaths()

# Загрузка основного конфига
with open(paths.get_config_path('config.json'), 'r') as f:
    config = json.load(f)

# Доступ к параметрам
pendulum_damping = config['pendulum']['damping']
spore_scale = config['spore']['scale']
debug_enabled = config['debug']['enable_verbose_output']
```

### **Использование ColorManager:**
```python
from src.managers.color_manager import ColorManager

color_manager = ColorManager()

# Получение цветов
spore_color = color_manager.get_color('spore', 'normal')
goal_color = color_manager.get_color('spore', 'goal')
ui_text_color = color_manager.get_color('ui', 'text')
```

### **Обновление конфигурации в реальном времени:**
Благодаря системе watcher (v13_manual), изменения в JSON файлах автоматически перезапускают симуляцию:

```bash
# 1. Запустите watcher
python src/utils/watcher.py

# 2. Отредактируйте config/json/config.json
# 3. Сохраните файл
# 4. Симуляция автоматически перезапустится с новыми настройками!
```

---

## 🔥 Практические примеры настройки

### **Включить отладочный вывод:**
```json
// config/json/config.json
{
  "debug": {
    "enable_verbose_output": true,
    "enable_detailed_evolution": true,
    "enable_candidate_logging": true,
    "enable_trajectory_logging": true
  }
}
```

### **Изменить физику маятника:**
```json
// config/json/config.json
{
  "pendulum": {
    "damping": 0.05,        // Уменьшить затухание (более колебательная система)
    "max_control": 2.0      // Увеличить максимальное управление
  }
}
```

### **Настроить цветовую схему:**
```json
// config/json/colors.json
{
  "spore": {
    "normal": [0, 1, 1, 1],     // Голубые обычные споры
    "goal": [1, 0, 1, 1],       // Пурпурная цель
    "preview": [1, 0.5, 0, 0.7] // Оранжевый превью
  }
}
```

### **Включить визуализацию стоимости:**
```json
// config/json/config.json
{
  "angel": {
    "show_angels": true,     // Включить "ангелов"
    "show_pillars": true,    // Показывать столбы
    "pillar_alpha": 0.9      // Более яркие столбы
  }
}
```

---

## 💡 Советы по настройке

### **Для разработки:**
- Включите все виды отладочного вывода
- Используйте watcher для мгновенного применения изменений
- Экспериментируйте с цветами для лучшей видимости компонентов

### **Для демонстрации:**
- Отключите отладочный вывод для чистого интерфейса
- Настройте приятную цветовую схему
- Включите визуализацию стоимости для наглядности

### **Для производительности:**
- Отключите ненужные визуальные элементы
- Уменьшите прозрачность и эффекты
- Выключите отладочный вывод

---

**🎯 Помните**: Все изменения в JSON файлах применяются автоматически благодаря системе watcher v13_manual!
# ▶️ Скрипты для запуска — v13_manual

Эта директория содержит исполняемые скрипты для запуска различных демонстраций проекта "Споры".

## 📁 Структура

```
scripts/
├── 📄 README.md           # Этот файл
└── 🚀 run/               # Исполняемые скрипты
    └── main_demo.py      # Главная демонстрация v13_manual
```

## 🎮 Основные скрипты

### **`run/main_demo.py`** — Главная демонстрация
> 🆕 **v13_manual**: Демонстрация системы ручного создания спор

**Что показывает:**
- ✨ Интерактивное создание спор с помощью мыши
- 🔮 Предсказания управления (min/max траектории)
- 🎯 Полупрозрачный превью курсора
- 🎨 Полный UI с инструкциями и метриками

**Особенности v13_manual:**
- Отключен автоматический SpawnArea
- Добавлен ManualSporeManager для ручного управления
- ЛКМ создает спору + 2 дочерние (min/max control)
- Клавиша C для полной очистки

## 🚀 Способы запуска

### 1. **Обычный запуск**
```bash
# Из корня проекта
python spores/v13_manual/scripts/run/main_demo.py

# Из директории v13_manual
cd spores/v13_manual
python scripts/run/main_demo.py
```

### 2. **Запуск с автоперезагрузкой** (для разработки)
```bash
# Автоматически перезапускается при изменении .py файлов
python src/utils/watcher.py main_demo.py

# Можно указать конкретный скрипт
python src/utils/watcher.py main_demo
```

### 3. **Запуск с отладкой**
```bash
# С включением всех видов отладочного вывода
# Отредактируйте config/json/config.json:
{
  "debug": {
    "enable_verbose_output": true,
    "enable_detailed_evolution": true,
    "enable_candidate_logging": true,
    "enable_trajectory_logging": true
  }
}
```

## ⚙️ Конфигурация запуска

### **Настройка монитора**
```python
# В main_demo.py, строка ~47
window_manager = WindowManager(monitor='down')  # 'main', 'top', 'right', 'down'
```

### **Включение/отключение компонентов**
```python
# В main_demo.py, строка ~54
USE_SPAWN_AREA = False  # True для включения автоматической области спавна
```

### **Настройка UI позиций**
UI автоматически адаптируется под выбранный монитор через `ui_constants.py`.

## 🎮 Интерактивное использование

### **После запуска main_demo.py:**

1. **🖱️ Ручное создание спор:**
   - Двигайте мышью — видите полупрозрачную спору-превью
   - Наблюдайте 2 призрака (предсказания min/max управления)
   - Кликните ЛКМ — создается настоящая спора + 2 дочерние

2. **🎯 Эволюция спор:**
   - Нажмите **F** — создать следующую спору от последней
   - Нажмите **G** — активировать случайного кандидата
   - Нажмите **V** — развить всех кандидатов до завершения

3. **📹 Навигация:**
   - **WASD** — движение камеры
   - **Space/Shift** — вверх/вниз
   - **Мышь** — поворот камеры
   - **Alt** — захват/освобождение курсора

4. **🔍 Масштабирование:**
   - **E/T** — приблизить/отдалить
   - **R** — сбросить зум
   - **1/2** — масштаб спор

5. **🎨 Визуализация:**
   - **Y** — включить/выключить ангелов (стоимость)
   - **H** — скрыть/показать весь UI
   - **C** — очистить все споры (v13_manual)

## 🐛 Устранение проблем

### **Ошибки при запуске:**

**ModuleNotFoundError:**
```bash
# Убедитесь что запускаете из правильной директории
cd spores/v13_manual
python scripts/run/main_demo.py
```

**Ошибки Ursina:**
```bash
# Проверьте установку зависимостей
pip install -r requirements.txt

# Или переустановите Ursina
pip uninstall ursina
pip install ursina
```

**Проблемы с производительностью:**
```python
# Отключите отладочный вывод в config/json/config.json
{
  "debug": {
    "enable_verbose_output": false,
    "enable_detailed_evolution": false,
    "enable_candidate_logging": false,
    "enable_trajectory_logging": false
  }
}
```

### **Настройка для разных систем:**

**Windows:**
```bash
# Используйте python вместо python3
python scripts/run/main_demo.py
```

**Linux/macOS:**
```bash
# Может потребоваться python3
python3 scripts/run/main_demo.py
```

**Виртуальное окружение:**
```bash
# Активируйте venv перед запуском
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
python scripts/run/main_demo.py
```

## 📊 Мониторинг производительности

Во время работы отслеживайте:
- **FPS** — в заголовке окна
- **Количество спор** — в левом верхнем углу
- **Использование памяти** — через Task Manager/htop

## 🔧 Создание собственных скриптов

### **Шаблон нового скрипта:**
```python
# scripts/run/my_demo.py
import sys
import os

# Настройка путей
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

from ursina import *
from src.core.spore import Spore
# ... другие импорты

app = Ursina()

# Ваша логика демонстрации

if __name__ == '__main__':
    app.run()
```

### **Интеграция с watcher.py:**
```bash
# Добавьте свой скрипт в watcher
python src/utils/watcher.py my_demo.py
```

---

**💡 Совет**: Начните с `main_demo.py` для понимания базовой функциональности, затем экспериментируйте с настройками в `config/json/` файлах.
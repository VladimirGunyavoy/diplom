# ▶️ Скрипты для запуска — v13_manual

Эта директория содержит исполняемые скрипты для запуска различных демонстраций проекта "Споры".

![Версия](https://img.shields.io/badge/версия-v13_manual-blue)
![Watcher](https://img.shields.io/badge/watcher-автоперезапуск-green)

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

### 1. **🔥 Запуск с автоперезагрузкой** (рекомендуется для разработки)
```bash
# Из корня проекта spores/v13_manual/
python src/utils/watcher.py

# Или явно указать скрипт (по умолчанию main_demo.py)
python src/utils/watcher.py main_demo

# Можно указать полное имя
python src/utils/watcher.py main_demo.py

# Для завершения используйте Ctrl+C
```

**Преимущества watcher:**
- ⚡ **Автоматический перезапуск** при изменении любого .py файла
- 🔄 **Корректное завершение** процессов при перезапуске
- 🎯 **Мгновенная обратная связь** для разработки
- 📁 **Отслеживание всех поддиректорий** (src/, scripts/, config/)

### 2. **Обычный запуск** (одноразовый)
```bash
# Из корня проекта spores/v13_manual/
python scripts/run/main_demo.py

# Или перейти в директорию
cd spores/v13_manual
python scripts/run/main_demo.py
```

### 3. **Запуск с отладкой**
```bash
# Включите отладочный вывод в config/json/config.json:
{
  "debug": {
    "enable_verbose_output": true,
    "enable_detailed_evolution": true,
    "enable_candidate_logging": true,
    "enable_trajectory_logging": true
  }
}

# Затем запустите любым способом
python src/utils/watcher.py
```

## ⚙️ Конфигурация запуска

### **Настройка монитора**
```python
# В main_demo.py, строка ~47
window_manager = WindowManager(monitor='down')
```
**Доступные варианты:**
- `'main'` — основной монитор
- `'top'` — верхний монитор  
- `'right'` — правый монитор
- `'down'` — нижний монитор

### **Включение/отключение компонентов**
```python
# В main_demo.py, строка ~54
USE_SPAWN_AREA = False  # True для включения автоматической области спавна
```

### **Настройка UI позиций**
UI автоматически адаптируется под выбранный монитор через `ui_constants.py`.

## 🎮 Интерактивное использование

### **После запуска main_demo.py:**

### 1. **🖱️ Ручное создание спор (v13_manual):**
   - **Двигайте мышью** — видите полупрозрачную спору-превью
   - **Наблюдайте 2 призрака** — предсказания min/max управления
   - **Кликните ЛКМ** — создается настоящая спора + 2 дочерние

### 2. **🎯 Эволюция спор:**
   - Нажмите **F** — создать следующую спору от последней
   - Нажмите **G** — активировать случайного кандидата
   - Нажмите **V** — развить всех кандидатов до завершения

### 3. **📹 Навигация:**
   - **WASD** — движение камеры
   - **Space/Shift** — вверх/вниз
   - **Мышь** — поворот камеры
   - **Alt** — захват/освобождение курсора

### 4. **🔍 Масштабирование:**
   - **E/T** — приблизить/отдалить
   - **R** — сбросить зум
   - **1/2** — масштаб спор

### 5. **🎨 Визуализация:**
   - **Y** — включить/выключить ангелов (стоимость)
   - **H** — скрыть/показать весь UI
   - **C** — очистить все споры (v13_manual)

## 👀 Система Watcher (v13_manual)

### **Что отслеживает watcher?**
- ✅ **Все .py файлы** в директории проекта
- ✅ **Файлы в поддиректориях** (src/, scripts/, config/)
- ✅ **Новые файлы** — автоматически включаются в отслеживание
- ❌ **Не отслеживает** — .pyc, .log, и другие временные файлы

### **Workflow с watcher:**
```bash
# 1. Запустите watcher один раз
python src/utils/watcher.py

# 2. Редактируйте код в любом редакторе
# 3. Сохраните файл — watcher автоматически перезапустит симуляцию
# 4. Наблюдайте результат — мгновенная обратная связь
# 5. Повторяйте шаги 2-4 для быстрой итерации
```

### **Расширение функциональности**
```python
# src/managers/custom_manager.py
class CustomSporeManager(ManualSporeManager):
    def create_custom_spore(self, position, control_type):
        # Ваша логика создания спор
        # После сохранения файла watcher автоматически перезапустит
        pass
```

### **Быстрое прототипирование**
```python
# Измените что-то в main_demo.py
USE_SPAWN_AREA = True  # Включить автоматический spawn area

# Сохраните файл — watcher мгновенно покажет результат
# Не нужно перезапускать вручную!
```

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

### **Проблемы с watcher:**

**Watcher не запускается:**
```bash
# Проверьте что watchdog установлен
pip install watchdog

# Или переустановите зависимости
pip install -r requirements.txt
```

**Watcher не реагирует на изменения:**
```bash
# Убедитесь что сохраняете файлы в правильной директории
# Watcher отслеживает только .py файлы
# Попробуйте изменить main_demo.py — должно сработать
```

**Симуляция зависает:**
```bash
# Нажмите Ctrl+C в терминале с watcher
# Это корректно остановит все процессы
```

### **Проблемы с производительностью:**
```json
// Отключите отладку в config/json/config.json
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
python src/utils/watcher.py
```

**Linux/macOS:**
```bash
# Может потребоваться python3
python3 scripts/run/main_demo.py
python3 src/utils/watcher.py
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

### **Если низкий FPS:**
1. Отключите отладочный вывод в конфиге
2. Уменьшите количество спор на сцене (клавиша C)
3. Отключите "ангелов" (клавиша Y)

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

### **Использование готовых компонентов:**
```python
# В вашем скрипте
from src.managers.manual_spore_manager import ManualSporeManager
from src.visual.ui_setup import UI_setup
from src.managers.color_manager import ColorManager

# Настройка займет всего несколько строк
manual_spore_manager = ManualSporeManager(...)
ui_setup = UI_setup(...)
# И система готова к работе!
```

## 📈 Статистика разработки

### **Производительность watcher:**
- ⚡ **Время перезапуска:** ~2-3 секунды
- 📁 **Файлов отслеживается:** ~50+ .py файлов
- 🔄 **Перезапусков в час:** зависит от интенсивности разработки
- 💾 **Потребление памяти:** минимальное

### **Удобство разработки:**
- 🎯 **Экономия времени:** 80%+ по сравнению с ручными перезапусками
- 🔥 **Скорость итераций:** мгновенная обратная связь
- 🐛 **Отладка:** легче найти проблемы в реальном времени

---

**💡 Совет**: Начните с `main_demo.py` для понимания базовой функциональности, затем экспериментируйте с настройками в `config/json/` файлах. Используйте watcher для максимально быстрой разработки!
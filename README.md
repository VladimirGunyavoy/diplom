# Проект "Споры" (версия 13 Manual)

> 🎮 Интерактивная 3D-симуляция для исследования алгоритмов управления и оптимизации с **ручным созданием спор**

![Статус проекта](https://img.shields.io/badge/status-stable-green)
![Версия](https://img.shields.io/badge/version-v13_manual-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Лицензия](https://img.shields.io/badge/license-MIT-green)

https://github.com/user-attachments/assets/574e9283-f366-4191-881b-5079a0229c13

## 🎯 Что нового в v13_manual?

### 🖱️ Система ручного создания спор
- **Интерактивный превью** — полупрозрачная спора следует за курсором мыши
- **Предсказания управления** — показ min/max траекторий в реальном времени  
- **Создание по клику** — ЛКМ создает родительскую спору + 2 дочерние (min/max control)
- **Отключен автоматический spawn area** — полный контроль пользователя

### 👀 Система автоматического перезапуска
- **Watcher система** — автоматический перезапуск при изменении кода
- **Мониторинг файлов** — отслеживает все .py файлы в проекте
- **Умное управление процессами** — корректное завершение и перезапуск
- **Режим разработки** — идеально для итеративной разработки

### 🎮 Новое управление
- **ЛКМ** — создать спору в позиции курсора с предсказанными детьми
- **C** — полная очистка всех спор и объектов  
- **Движение мыши** — автоматическое обновление превью и предсказаний

## 🚀 Основные технологии

-   **Движок:** `Ursina Engine` — для создания 3D-сцены и обработки пользовательского ввода
-   **Научные вычисления:** `NumPy` и `SciPy` — для всех математических расчетов
-   **Визуализация данных:** `Matplotlib` — для генерации линий уровня на поверхности стоимости
-   **Алгоритмы размещения:** `Poisson Disk Sampling` — для равномерной генерации кандидатских спор
-   **Система разработки:** `Watchdog` — автоматический перезапуск при изменениях

## 🛠️ Быстрый старт

### 1. Требования
- **Python 3.9+**
- **Git** для клонирования репозитория

### 2. Установка
```bash
# Клонируйте репозиторий
git clone <URL репозитория>
cd spores/v13_manual

# Установите зависимости
pip install -r requirements.txt
```

### 3. Запуск для разработки (рекомендуется)
```bash
# 🔥 Автоматический перезапуск при изменении кода
python src/utils/watcher.py

# Или явно указать скрипт (по умолчанию main_demo.py)
python src/utils/watcher.py main_demo

# Для завершения используйте Ctrl+C
```

### 4. Обычный запуск (без автоперезапуска)
```bash
# Одноразовый запуск
python scripts/run/main_demo.py
```

### 5. Первые шаги
1. **Запустите watcher** — увидите автозапуск main_demo.py
2. **Подвигайте мышью** — спора-превью будет следовать за курсором  
3. **Кликните ЛКМ** — создастся реальная спора + 2 дочерние
4. **Измените код** — watcher автоматически перезапустит симуляцию
5. **Используйте WASD** для навигации по сцене

## 🎮 Управление

### 🖱️ Ручное создание спор (v13_manual)
- **ЛКМ** — создать спору + детей в позиции курсора
- **Движение мыши** — обновить превью и предсказания
- **C** — очистить все споры

### 🎯 Базовое управление спорами
- **F** — создать новую спору от последней активной
- **G** — активировать случайного кандидата
- **V** — развить всех кандидатов до завершения

### 📹 Камера и навигация
- **WASD** — движение камеры
- **Space/Shift** — подъем/спуск камеры  
- **Мышь** — поворот камеры
- **Alt** — захват/освобождение курсора

### 🔍 Масштабирование
- **E/T** — приблизить/отдалить сцену
- **R** — сбросить масштаб
- **1/2** — изменить масштаб спор

### 🎛️ Настройки отображения
- **Y** — переключить отображение ангелов (стоимость)
- **H** — скрыть/показать весь UI
- **5/6** — уменьшить/увеличить радиус генерации кандидатов

### ⚙️ Системные команды
- **Q/Escape** — выход из приложения
- **Ctrl+C** — остановить watcher (в терминале)

## 👀 Watcher система — автоматический перезапуск

### 🔥 Почему использовать watcher?
- **Мгновенная обратная связь** — видите изменения сразу после сохранения
- **Ускорение разработки** — не нужно вручную перезапускать скрипт
- **Отслеживание всех файлов** — следит за изменениями во всем проекте
- **Умное управление** — корректно завершает старые процессы

### 🎯 Основные команды watcher
```bash
# Запуск по умолчанию (main_demo.py)
python src/utils/watcher.py

# Запуск конкретного скрипта
python src/utils/watcher.py my_script

# Можно указать с расширением или без
python src/utils/watcher.py main_demo.py
```

### ⚡ Workflow разработки
1. **Запустите watcher** один раз
2. **Редактируйте код** в любом редакторе
3. **Сохраните файл** — watcher автоматически перезапустит симуляцию
4. **Наблюдайте результат** — мгновенная обратная связь
5. **Повторяйте** шаги 2-4 для быстрой итерации

### 🛠️ Что отслеживает watcher?
- ✅ **Все .py файлы** в директории проекта
- ✅ **Файлы в поддиректориях** (src/, scripts/, config/)
- ✅ **Новые файлы** — автоматически включаются в отслеживание
- ❌ **Не отслеживает** — .pyc, .log, и другие временные файлы

### 🔧 Устранение проблем watcher
```bash
# Если процесс завис
Ctrl+C  # Корректно остановит watcher и дочерние процессы

# Если скрипт не найден
python src/utils/watcher.py
# Покажет список доступных скриптов в scripts/run/

# Если watcher не реагирует на изменения
# Проверьте что файл действительно сохранился
# и находится в отслеживаемой директории
```

## 🏛️ Архитектура v13_manual

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

## 📁 Структура проекта

```
spores/v13_manual/
├── 📄 README.md                 # Этот файл
├── 📦 requirements.txt          # Зависимости Python
├── 🎮 scripts/run/             # Исполняемые скрипты
│   └── main_demo.py            # Главная демонстрация
├── 💻 src/                     # Исходный код
│   ├── core/                   # Интеграционный слой (Spore)
│   ├── logic/                  # Математика и алгоритмы
│   ├── visual/                 # 3D визуализация и UI
│   ├── managers/               # Менеджеры подсистем
│   │   └── manual_spore_manager.py  # 🆕 Ручное создание
│   └── utils/                  # Утилиты
│       └── watcher.py          # 👀 Автоперезапуск
├── ⚙️ config/                  # Конфигурационные файлы
│   └── json/                   # JSON конфиги
├── 📚 theory/                  # Математическая теория
├── 📊 changes/                 # Логи изменений v13_manual
└── 🎨 assets/                  # 3D модели и ресурсы
```

## 🎮 Примеры использования

### Разработка с watcher
```bash
# Терминал 1: Запустите watcher
python src/utils/watcher.py

# Терминал 2: Откройте любой редактор кода
code src/managers/manual_spore_manager.py

# Внесите изменения и сохраните
# Watcher автоматически перезапустит симуляцию
# Наблюдайте результат в 3D окне
```

### Создание спор с предсказаниями
1. Запустите watcher: `python src/utils/watcher.py`
2. Наведите мышь на интересную позицию  
3. Посмотрите на предсказания (2 призрака показывают min/max управление)
4. Кликните ЛКМ для создания реальных спор

### Исследование траекторий
1. Создайте несколько спор в разных местах
2. Используйте **F** для продолжения эволюции от последней споры
3. Наблюдайте как споры ищут оптимальные пути к цели
4. Измените параметры в `config/json/config.json` и сохраните
5. Watcher перезапустит симуляцию с новыми параметрами

### Анализ функции стоимости  
1. Нажмите **Y** для показа "ангелов" (визуализация стоимости)
2. Споры на возвышениях показывают области высокой стоимости
3. Цель всегда в самой низкой точке

## ⚙️ Конфигурация

### 📊 Настройка отладочного вывода
```json
// config/json/config.json
{
  "debug": {
    "enable_verbose_output": true,       // Общие сообщения
    "enable_detailed_evolution": true,   // Детали эволюции
    "enable_candidate_logging": true,    // Кандидатские споры
    "enable_trajectory_logging": true    // Траектории
  }
}
```

### 🎮 Настройка симуляции
```json
// config/json/config.json  
{
  "pendulum": {
    "damping": 0.1,              // Затухание системы
    "max_control": 1.0           // Максимальное управление
  },
  "spore": {
    "scale": 0.05,              // Размер спор
    "goal_position": [3.14159, 0] // Позиция цели
  }
}
```

### 🖥️ Настройка мониторов
```python
# scripts/run/main_demo.py, строка ~47
window_manager = WindowManager(monitor='down')
# Варианты: 'main', 'top', 'right', 'down'
```

## 📚 Документация

- **[💻 Исходный код](src/README.md)** — архитектура и структура кода v13_manual
- **[🎮 Скрипты](scripts/README.md)** — запуск и использование watcher
- **[📐 Теория](theory/README.md)** — математические основы проекта
- **[📝 Изменения](changes/logs/)** — логи разработки v13_manual

## 🔬 Для разработчиков

### Workflow с watcher
```bash
# 1. Запустите watcher один раз
python src/utils/watcher.py

# 2. Редактируйте код в любом редакторе
# 3. Сохраните файл — автоматический перезапуск
# 4. Наблюдайте результат в 3D окне
# 5. Повторяйте шаги 2-4
```

### Расширение функциональности
```python
# src/managers/custom_manager.py
class CustomSporeManager(ManualSporeManager):
    def create_custom_spore(self, position, control_type):
        # Ваша логика создания спор
        # После сохранения файла watcher автоматически перезапустит
        pass
```

### Быстрое прототипирование
```python
# Измените что-то в main_demo.py
USE_SPAWN_AREA = True  # Включить автоматический spawn area

# Сохраните файл — watcher мгновенно покажет результат
# Не нужно перезапускать вручную!
```

## 🔧 Устранение проблем

### Watcher не запускается
```bash
# Проверьте что watchdog установлен
pip install watchdog

# Или переустановите зависимости
pip install -r requirements.txt
```

### Watcher не реагирует на изменения
```bash
# Убедитесь что сохраняете файлы в правильной директории
# Watcher отслеживает только .py файлы
# Попробуйте изменить main_demo.py — должно сработать
```

### Симуляция зависает
```bash
# Нажмите Ctrl+C в терминале с watcher
# Это корректно остановит все процессы
```

### Низкая производительность
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

## 🤝 Участие в разработке

1. **Fork** репозитория
2. **Запустите watcher**: `python src/utils/watcher.py`
3. **Создайте feature branch**: `git checkout -b feature/amazing-feature`
4. **Разрабатывайте** с автоматическим перезапуском
5. **Commit** изменения: `git commit -m 'Add amazing feature'`
6. **Push** в branch: `git push origin feature/amazing-feature`
7. Откройте **Pull Request**

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл `LICENSE` для деталей.

## 📬 Контакты

- 🐛 **Баги и предложения**: [Issues](../../issues)
- 💬 **Обсуждения**: [Discussions](../../discussions)  
- 📧 **Email**: your.email@example.com

---

<div align="center">

**⭐ Если проект вам понравился, поставьте звезду! ⭐**

*Создано с ❤️ для исследования алгоритмов управления*

**🚀 Попробуйте watcher — это изменит ваш workflow разработки!**

</div>
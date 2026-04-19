# ⚙️ Конфигурация — v14_back

Эта директория и связанные с ней модули отвечают за всю конфигурацию проекта с новым акцентом на временные параметры и обратную динамику.

![Версия](https://img.shields.io/badge/версия-v14_back-blue)
![Конфигурация](https://img.shields.io/badge/config-JSON%2BPython%2BTemporal-orange)
![Временная система](https://img.shields.io/badge/time_config-backward_dynamics-green)

---

## 📄 JSON-конфигурация (`json/`)

Расширенная система конфигурации для поддержки временных операций и обратной динамики.

### `config.json` — Основной конфигурационный файл (обновлен для v14_back)

```json
{
  "spore": {
    "scale": 0.1,
    "goal_position": [0, 0],
    "max_evolution_steps": 1000,
    "temporal_tracking": true
  },
  
  "pendulum": {
    "damping": 0.1,
    "control_bounds": [-2.0, 2.0],
    "dt": 0.1,
    "backward_dynamics_enabled": true,
    "reverse_integration_method": "rk45"
  },
  
  "time_system": {
    "max_history_size": 1000,
    "auto_bookmark_interval": 100,
    "compression_enabled": true,
    "interpolation_method": "cubic_spline",
    "backward_step_size": 0.1,
    "performance_monitoring": false
  },
  
  "debug": {
    "general": false,
    "evolution": false,
    "candidates": false,
    "trajectory": false,
    "temporal_operations": false,
    "state_transitions": false,
    "backward_dynamics": false,
    "performance_tracking": false
  }
}
```

**Новые секции v14_back:**

#### `time_system` — Конфигурация временной системы
- `max_history_size`: Максимальное количество сохраняемых состояний
- `auto_bookmark_interval`: Интервал автоматического создания временных меток
- `compression_enabled`: Сжатие старых состояний для экономии памяти
- `interpolation_method`: Метод интерполяции между состояниями (`linear`, `cubic_spline`)
- `backward_step_size`: Размер шага при движении назад во времени
- `performance_monitoring`: Включение мониторинга производительности

#### `debug` — Расширенная отладка (новые флаги)
- `temporal_operations`: Отладка операций с временем
- `state_transitions`: Отладка переходов между состояниями
- `backward_dynamics`: Отладка обратной динамики
- `performance_tracking`: Отслеживание производительности временных операций

### `temporal_config.json` — Специализированная конфигурация времени (новое в v14_back)

```json
{
  "backward_dynamics": {
    "enabled": true,
    "integration_method": "rk45",
    "error_tolerance": 1e-6,
    "max_iterations": 1000,
    "adaptive_step_size": true
  },
  
  "state_history": {
    "storage_method": "compressed",
    "compression_ratio": 0.8,
    "cleanup_threshold": 2000,
    "persistence_enabled": true,
    "persistence_file": "temp/state_history.dat"
  },
  
  "trajectory_analysis": {
    "enabled": true,
    "analysis_depth": 100,
    "path_optimization": true,
    "branching_factor": 3,
    "convergence_threshold": 0.01
  },
  
  "performance": {
    "profiling_enabled": false,
    "memory_optimization": true,
    "lazy_computation": true,
    "parallel_processing": false,
    "cache_size": 500
  },
  
  "ui_temporal": {
    "time_scrubber_enabled": true,
    "timeline_resolution": 100,
    "smooth_transitions": true,
    "preview_depth": 10,
    "ghost_trail_length": 20
  }
}
```

### `performance_config.json` — Конфигурация производительности (новое в v14_back)

```json
{
  "memory_management": {
    "auto_cleanup_enabled": true,
    "cleanup_interval": 1000,
    "memory_threshold_mb": 512,
    "garbage_collection_frequency": 100
  },
  
  "computational_optimization": {
    "vectorized_operations": true,
    "numba_acceleration": false,
    "multiprocessing_enabled": false,
    "thread_pool_size": 4
  },
  
  "caching": {
    "state_cache_size": 200,
    "computation_cache_size": 100,
    "trajectory_cache_size": 50,
    "cache_expiry_time": 3600
  },
  
  "profiling": {
    "enabled": false,
    "output_file": "logs/performance_profile.txt",
    "detailed_timing": false,
    "memory_tracking": false
  }
}
```

## 🐍 Python-конфигурация (`../src/config/`)

Системные настройки, связанные с кодом и новой временной архитектурой.

### `temporal_config.py` — Конфигурация временной системы (новое в v14_back)

```python
"""
Конфигурация временной системы v14_back
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class TemporalConfig:
    """Конфигурация временных операций"""
    
    # Основные параметры времени
    max_history_size: int = 1000
    auto_bookmark_interval: int = 100
    backward_step_size: float = 0.1
    
    # Настройки обратной динамики
    backward_integration_method: str = "rk45"
    backward_error_tolerance: float = 1e-6
    backward_max_iterations: int = 1000
    
    # Оптимизация производительности
    compression_enabled: bool = True
    lazy_computation: bool = True
    parallel_processing: bool = False
    
    # Интерполяция и сглаживание
    interpolation_method: str = "cubic_spline"
    smooth_transitions: bool = True
    
    @classmethod
    def from_json(cls, config_dict: Dict[str, Any]) -> 'TemporalConfig':
        """Создание конфигурации из JSON"""
        return cls(**config_dict.get('time_system', {}))

@dataclass 
class BackwardDynamicsConfig:
    """Специализированная конфигурация обратной динамики"""
    
    enabled: bool = True
    integration_method: str = "rk45"
    error_tolerance: float = 1e-6
    max_iterations: int = 1000
    adaptive_step_size: bool = True
    
    # Продвинутые настройки
    jacobian_computation: str = "automatic"  # "automatic", "finite_difference", "analytical"
    stability_check: bool = True
    convergence_acceleration: bool = False
```

## 🔧 Руководство по конфигурации v14_back

### **Настройка для разных сценариев использования:**

#### 🚀 **Высокая производительность** (для реального времени)
```json
{
  "time_system": {
    "max_history_size": 200,
    "compression_enabled": true,
    "performance_monitoring": false
  },
  "performance": {
    "memory_optimization": true,
    "lazy_computation": true,
    "vectorized_operations": true
  }
}
```

#### 🔬 **Детальный анализ** (для исследований)
```json
{
  "time_system": {
    "max_history_size": 5000,
    "auto_bookmark_interval": 50,
    "interpolation_method": "cubic_spline"
  },
  "trajectory_analysis": {
    "analysis_depth": 500,
    "path_optimization": true
  }
}
```

#### 🐛 **Отладка временной системы**
```json
{
  "debug": {
    "temporal_operations": true,
    "state_transitions": true,
    "backward_dynamics": true,
    "performance_tracking": true
  },
  "performance": {
    "profiling_enabled": true,
    "detailed_timing": true,
    "memory_tracking": true
  }
}
```

#### 💾 **Ограниченная память** (для слабых систем)
```json
{
  "time_system": {
    "max_history_size": 100,
    "compression_enabled": true
  },
  "memory_management": {
    "auto_cleanup_enabled": true,
    "memory_threshold_mb": 256,
    "cleanup_interval": 500
  }
}
```

### **Настройка обратной динамики:**

#### ⚡ **Быстрая обратная динамика** (приблизительная)
```json
{
  "backward_dynamics": {
    "integration_method": "euler",
    "error_tolerance": 1e-3,
    "adaptive_step_size": false
  }
}
```

#### 🎯 **Точная обратная динамика** (высокая точность)
```json
{
  "backward_dynamics": {
    "integration_method": "rk45",
    "error_tolerance": 1e-9,
    "adaptive_step_size": true,
    "max_iterations": 5000
  }
}
```

### **Настройка UI временной системы:**

#### 📊 **Богатая временная визуализация**
```json
{
  "ui_temporal": {
    "time_scrubber_enabled": true,
    "timeline_resolution": 200,
    "ghost_trail_length": 50,
    "smooth_transitions": true
  }
}
```

#### 🎮 **Минималистичный интерфейс**
```json
{
  "ui_temporal": {
    "time_scrubber_enabled": false,
    "timeline_resolution": 50,
    "ghost_trail_length": 5,
    "smooth_transitions": false
  }
}
```

## 📊 Мониторинг и профилирование

### **Включение мониторинга производительности:**
```json
{
  "time_system": {
    "performance_monitoring": true
  },
  "debug": {
    "performance_tracking": true
  },
  "profiling": {
    "enabled": true,
    "output_file": "logs/temporal_performance.log",
    "detailed_timing": true
  }
}
```

### **Анализ результатов:**
```bash
# Просмотр отчета о производительности
cat logs/temporal_performance.log

# Анализ узких мест
grep "SLOW" logs/temporal_performance.log

# Мониторинг использования памяти
grep "MEMORY" logs/temporal_performance.log
```

## 🔄 Миграция конфигурации с v13_manual

### **Автоматическая миграция:**
```python
# Скрипт миграции (включен в v14_back)
python scripts/migrate_config_v13_to_v14.py

# Проверка совместимости
python scripts/validate_config.py config/json/config.json
```

### **Ручная миграция основных настроек:**
```json
// v13_manual → v14_back
{
  // Старое (v13_manual)
  "spore": {
    "scale": 0.1
  },
  
  // Новое (v14_back) - добавляем временное отслеживание
  "spore": {
    "scale": 0.1,
    "temporal_tracking": true
  }
}
```

### **Новые обязательные параметры v14_back:**
```json
{
  "time_system": {
    "max_history_size": 1000,        // ОБЯЗАТЕЛЬНО
    "backward_step_size": 0.1        // ОБЯЗАТЕЛЬНО
  },
  "pendulum": {
    "backward_dynamics_enabled": true // ОБЯЗАТЕЛЬНО
  }
}
```

## 🎯 Лучшие практики конфигурации v14_back

### **Оптимизация для продуктивного использования:**
1. **Ограничьте размер истории** — установите разумный `max_history_size`
2. **Включите сжатие** — `compression_enabled: true` для экономии памяти
3. **Отключите отладку** — все `debug` флаги в `false`
4. **Используйте ленивые вычисления** — `lazy_computation: true`

### **Настройка для разработки:**
1. **Включите нужную отладку** — только необходимые `debug` флаги
2. **Уменьшите размер истории** — для быстрой итерации
3. **Включите профилирование** — для оптимизации алгоритмов
4. **Сохраните состояния** — `persistence_enabled: true`

### **Конфигурация для исследований:**
1. **Максимальная история** — большой `max_history_size`
2. **Высокая точность** — строгие `error_tolerance`
3. **Детальный анализ** — включите `trajectory_analysis`
4. **Сохранение данных** — настройте `persistence`

---

**🔧 Начните с базовой конфигурации в `config.json`, затем тонко настройте временные параметры в `temporal_config.json` под ваши конкретные задачи!**
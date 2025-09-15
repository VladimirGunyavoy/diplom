from ursina import Vec2
import json
import os

"""
Централизованные константы для позиционирования элементов UI.
Использование: from .ui_constants import UI_POSITIONS
"""

class UI_POSITIONS:
    # --- Базовые позиции (для монитора main) ---
    _BASE_POSITIONS = {
        # --- Левая сторона экрана ---
        'GAME_CONTROLS': Vec2(-0.76, 0.46),
        'SPORE_INSTRUCTION': Vec2(-0.76, 0.19),
        'SPORE_COUNTER': Vec2(-0.76, 0.13),
        'DT_INFO': Vec2(-0.76, 0.07),
        
        # --- Правая сторона экрана ---
        'SPAWN_AREA_INFO': Vec2(0.5, 0.27),
        'SPAWN_AREA_INSTRUCTIONS': Vec2(0.5, 0.21),
        'LOOK_POINT_DEBUG': Vec2(0.5, 0.47),
        'PARAM_VALUE': Vec2(0.57, -0.31),
        'CANDIDATE_INFO': Vec2(0.29, 0.47),
        
        # --- Низ/Центр экрана ---
        'POSITION_INFO': Vec2(0.5, -0.44),
        'CURSOR_STATUS': Vec2(0, -0.45),
        
        # --- Позиции по умолчанию для UIManager ---
        'DEFAULT_INSTRUCTIONS': Vec2(-0.75, 0.45),
        'DEFAULT_COUNTER': Vec2(-0.95, 0.4),
        'DEFAULT_INFO_BLOCK': Vec2(-0.95, 0.9),
        'DEFAULT_DEBUG_INFO': Vec2(0.5, 0.45)
    }
    
    _current_monitor = "main"
    _ui_offsets = {}
    
    @classmethod
    def set_monitor(cls, monitor: str):
        """Устанавливает текущий монитор и загружает соответствующие отступы."""
        cls._current_monitor = monitor
        cls._load_ui_offsets()
        cls._apply_offsets()
    
    @classmethod
    def _load_ui_offsets(cls):
        """Загружает отступы для UI из конфигурационного файла."""
        try:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                '..', '..', 'config', 'json', 'ui_offsets.json'
            )
            with open(config_path, 'r', encoding='utf-8') as f:
                cls._ui_offsets = json.load(f)
        except Exception as e:
            print(f"Не удалось загрузить ui_offsets.json: {e}")
            cls._ui_offsets = {}
    
    @classmethod
    def _apply_offsets(cls):
        """Применяет отступы к позициям UI элементов."""
        if cls._current_monitor not in cls._ui_offsets:
            return
        
        offsets = cls._ui_offsets[cls._current_monitor]
        
        for key, base_pos in cls._BASE_POSITIONS.items():
            # Определяем тип отступа в зависимости от позиции элемента
            x_offset = 0
            y_offset = 0
            
            # Левая сторона экрана
            if base_pos.x < -0.5:
                x_offset = offsets.get('left_offset', 0)
            # Правая сторона экрана  
            elif base_pos.x > 0.2:
                x_offset = offsets.get('right_offset', 0)
            
            # Верх экрана
            if base_pos.y > 0.2:
                y_offset = offsets.get('top_offset', 0)
            # Низ экрана
            elif base_pos.y < -0.2:
                y_offset = offsets.get('bottom_offset', 0)
            
            # Применяем отступы
            new_pos = Vec2(base_pos.x + x_offset, base_pos.y + y_offset)
            setattr(cls, key, new_pos)

    # Инициализируем базовые позиции
    @classmethod 
    def _init_positions(cls):
        """Инициализирует позиции значениями по умолчанию."""
        for key, pos in cls._BASE_POSITIONS.items():
            setattr(cls, key, pos)

# Инициализируем позиции при импорте
UI_POSITIONS._init_positions()
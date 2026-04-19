from ursina import Vec2

"""
Централизованные константы для позиционирования элементов UI.
Использование: from .ui_constants import UI_POSITIONS
"""

class UI_POSITIONS:
    # --- Левая сторона экрана ---
    GAME_CONTROLS           = Vec2(-0.76, 0.46)
    SPORE_COUNTER           = Vec2(-0.76, 0.2)
    SPORE_INSTRUCTION       = Vec2(-0.76, 0.27)
    
    # --- Правая сторона экрана ---
    SPAWN_AREA_INFO         = Vec2(0.55, 0.3)
    SPAWN_AREA_INSTRUCTIONS = Vec2(0.5, 0.21)
    LOOK_POINT_DEBUG        = Vec2(0.5, 0.47)
    PARAM_VALUE             = Vec2(0.57, -0.31)
    
    # --- Низ/Центр экрана ---
    POSITION_INFO           = Vec2(0.4, -0.4)
    CURSOR_STATUS           = Vec2(0, -0.45)
    
    # --- Позиции по умолчанию для UIManager ---
    DEFAULT_INSTRUCTIONS    = Vec2(-0.75, 0.45)
    DEFAULT_COUNTER         = Vec2(-0.95, 0.4)
    DEFAULT_INFO_BLOCK      = Vec2(-0.95, 0.9)
    DEFAULT_DEBUG_INFO      = Vec2(0.5, 0.45) 
from ursina import (
    Entity, DirectionalLight, AmbientLight, color, window, mouse, held_keys, application, Light
)
from ursina.prefabs.first_person_controller import FirstPersonController # Import FirstPersonController
import numpy as np
import time
from typing import Optional, List, Tuple

from ..utils.scalable import Scalable
from ..managers.color_manager import ColorManager
from ..visual.ui_manager import UIManager

# Предварительное объявление
class VisualManager: pass

# Class for setting up scene, camera and lighting
class SceneSetup:
    def __init__(self, 
                 init_position: Tuple[float, float, float] = (1.5, -1, -2), 
                 init_rotation_x: float = 21, 
                 init_rotation_y: float = -35, 
                 color_manager: Optional[ColorManager] = None, 
                 ui_manager: Optional[UIManager] = None, 
                 **kwargs):
        # Используем переданный ColorManager или создаем новый
        self.color_manager: ColorManager = color_manager if color_manager is not None else ColorManager()
        
        # Используем переданный UIManager или создаем новый
        self.ui_manager: UIManager = ui_manager if ui_manager is not None else UIManager(self.color_manager)
        
        self.lights: List[Light] = [  # Create moderate lighting
            # DirectionalLight(rotation=(45, -45, 45), color=color.white, intensity=0.8),
            DirectionalLight(rotation=(-45, -45, 45), color=self.color_manager.get_color('scene', 'directional_light'), intensity=1.5),  # Увеличена интенсивность, Y и Z инвертированы
            DirectionalLight(rotation=(45, 0, 0), color=self.color_manager.get_color('scene', 'directional_light'), intensity=1.2),  # Дополнительный свет сверху
            AmbientLight(color=self.color_manager.get_color('scene', 'ambient_light'))  # Усилено окружающее освещение
        ]

        self.base_position: Tuple[float, float, float] = init_position
        
        # self.floor = Scalable(  # Create a plane under objects
        #     model='plane',
        #     color=self.color_manager.get_color('scene', 'floor'),
        #     scale=25,
        #     position=(0, 0, 0),
        #     texture='white_cube',
        #     texture_scale=(50, 50),
        #     double_sided=True
        # )
        
        self.base_speed: float = 2

        self.player: FirstPersonController = FirstPersonController(
            gravity=0,
            position=init_position,
            speed=self.base_speed
        )
        
        self.player.camera_pivot.rotation_x = init_rotation_x
        self.player.rotation_y = init_rotation_y
        
        # Новый флаг для "заморозки" ввода
        self.input_frozen: bool = False
        
        self.cursor_locked: bool = True
        mouse.locked = self.cursor_locked
        
        window.color = self.color_manager.get_color('scene', 'window_background')
        
        # Add flag for tracking cursor state
        self.cursor_locked = True
        
        # UI теперь полностью управляется через UI_setup, не создаем здесь
        
        # Управление UI переместилось в UI_setup
        
    def update_position_info(self) -> None:
        """Обновляется автоматически через UI Manager"""
        # Эта функция теперь вызывается автоматически через ui_manager.update_dynamic_elements()
        pass

    def toggle_freeze(self) -> None:
        """Переключает режим 'заморозки' всего ввода."""
        self.input_frozen = not self.input_frozen
        
        # Блокируем/разблокируем мышь и игрока
        mouse.locked = not self.input_frozen
        self.player.enabled = not self.input_frozen
        
        print(f"[SceneSetup] toggle_freeze called. New state: input_frozen={self.input_frozen}")

    def update(self, dt: float) -> None:
        """Updates additional parameters not included in FirstPersonController"""
        # Эта проверка должна выполняться всегда, чтобы можно было "разморозить" ввод
        if held_keys['alt'] and not hasattr(self, 'alt_timer'):
            self.alt_timer = time.time()
        
        if not held_keys['alt'] and hasattr(self, 'alt_timer'):
            if time.time() - self.alt_timer < 0.5:
                self.toggle_freeze()
            delattr(self, 'alt_timer')
            
        # Если ввод заморожен, остальную логику (движение) не выполняем
        if self.input_frozen:
            return
            
        self.player.y += (held_keys['space'] - held_keys['shift']) * self.player.speed * dt
        
        # Обновление UI переместилось в UI_setup
    
    def input_handler(self, key: str) -> None:
        """Input handler for program closing and speed control"""
        if key == 'q':
            application.quit()
        
        # Временное решение для переключения призраков
        if key == '7':
            if hasattr(self, 'visual_manager') and hasattr(self.visual_manager, 'ghost_visualizer'):
                self.visual_manager.ghost_visualizer.enabled = not self.visual_manager.ghost_visualizer.enabled

    def set_visual_manager(self, visual_manager: 'VisualManager') -> None:
        """Позволяет передать visual_manager после инициализации."""
        self.visual_manager = visual_manager

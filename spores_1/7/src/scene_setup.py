from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController # Import FirstPersonController
import numpy as np
import time

from src.scalable import Scalable
from src.color_manager import ColorManager
from src.ui_manager import UIManager

# Class for setting up scene, camera and lighting
class SceneSetup:
    def __init__(self, init_position=(1.5, -1, -2), init_rotation_x=21, init_rotation_y=-35, color_manager=None, ui_manager=None):
        # Используем переданный ColorManager или создаем новый
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager
        
        # Используем переданный UIManager или создаем новый
        if ui_manager is None:
            ui_manager = UIManager(color_manager)
        self.ui_manager = ui_manager
        
        self.lights = [  # Create moderate lighting
            # DirectionalLight(rotation=(45, -45, 45), color=color.white, intensity=0.8),
            DirectionalLight(rotation=(-45, -45, 45), color=self.color_manager.get_color('scene', 'directional_light'), intensity=1.5),  # Увеличена интенсивность, Y и Z инвертированы
            DirectionalLight(rotation=(45, 0, 0), color=self.color_manager.get_color('scene', 'directional_light'), intensity=1.2),  # Дополнительный свет сверху
            AmbientLight(color=self.color_manager.get_color('scene', 'ambient_light'))  # Усилено окружающее освещение
        ]

        self.base_position = init_position
        
        self.floor = Scalable(  # Create a plane under objects
            model='plane',
            color=self.color_manager.get_color('scene', 'floor'),
            scale=25,
            position=(0, 0, 0),
            texture='white_cube',
            texture_scale=(50, 50),
            double_sided=True
        )
        
        self.base_speed = 2
        self.player = FirstPersonController(  # Create first-person controller instead of EditorCamera
            gravity=0,  # Disable gravity for free movement
            position=init_position,  # New starting position
            speed=self.base_speed  # Movement speed
        )
        
        invoke(lambda: setattr(self.player.camera_pivot, 'rotation_x', init_rotation_x), delay=0.01)
        invoke(lambda: setattr(self.player, 'rotation_y', init_rotation_y), delay=0.01)
        
        window.color = self.color_manager.get_color('scene', 'window_background')
        
        # Add flag for tracking cursor state
        self.cursor_locked = True
        
        # UI теперь полностью управляется через UI_setup, не создаем здесь
        
        # Управление UI переместилось в UI_setup
        
    def update_position_info(self):
        """Обновляется автоматически через UI Manager"""
        # Эта функция теперь вызывается автоматически через ui_manager.update_dynamic_elements()
        pass


        
    def update(self, dt):
        """Updates additional parameters not included in FirstPersonController"""
        self.player.y += (held_keys['space'] - held_keys['shift']) * self.player.speed * dt  # Vertical movement (up/down)
        
        # Обработка клавиш UI переместилась в UI_setup
            
        if held_keys['alt'] and not hasattr(self, 'alt_timer'):  # Handle Alt press for unlocking/locking cursor
            self.alt_timer = time.time()
        
        if not held_keys['alt'] and hasattr(self, 'alt_timer'):
            if time.time() - self.alt_timer < 0.5:  # Check that it was a short press
                self.toggle_cursor_lock()  # If Alt was pressed and released
            delattr(self, 'alt_timer')
            
        # Обновление UI переместилось в UI_setup
    
    def toggle_cursor_lock(self):
        """Toggles cursor lock"""
        self.cursor_locked = not self.cursor_locked
        
        if self.cursor_locked:
            mouse.locked = True
            self.player.enabled = True
        else:
            mouse.locked = False
            self.player.enabled = False  # Disable the controller so the camera doesn't rotate
        
        # UI обновление будет происходить в UI_setup через функции обновления
    
    def input_handler(self, key):
        """Input handler for program closing and speed control"""
        if key == 'q':
            application.quit()

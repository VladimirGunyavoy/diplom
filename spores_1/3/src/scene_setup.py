from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController # Import FirstPersonController
import numpy as np
import time

from src.scalable import Scalable

# Class for setting up scene, camera and lighting
class SceneSetup:
    def __init__(self, init_position=(1.5, -1, -2), init_rotation_x=21, init_rotation_y=-35):
        self.lights = [  # Create moderate lighting
            # DirectionalLight(rotation=(45, -45, 45), color=color.white, intensity=0.8),
            DirectionalLight(rotation=(-45, -45, 45), color=color.white, intensity=1.5),  # Увеличена интенсивность, Y и Z инвертированы
            DirectionalLight(rotation=(45, 0, 0), color=color.white, intensity=1.2),  # Дополнительный свет сверху
            AmbientLight(color=color.rgba(0.6, 0.6, 0.65, 1))  # Усилено окружающее освещение
        ]

        self.base_position = init_position
        
        self.floor = Scalable(  # Create a plane under objects
            model='plane',
            color=color.rgb(0.3, 0.3, 0.4),  # Светлее пол для контраста
            scale=25,
            position=(0, 0, 0),
            texture='white_cube',
            texture_scale=(50, 50),
            alpha=0.5,
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
        
        self.position_info = Text(
            text="",
            position=(0.4, -0.4, 0),
            scale=0.7,
            color=color.white,
            background=True,
            font='VeraMono.ttf',
        )
        
        window.color = color.rgb(0.08, 0.1, 0.2)  # Светлее фон, как и выше
        
        # Add flag for tracking cursor state
        self.cursor_locked = True
        
        self.cursor_status = Text(  # Add cursor state indicator
            text="Cursor locked [Alt to unlock]",
            position=(0, -0.45),
            scale=0.7,
            color=color.rgba(1, 1, 1, 0.7),
            origin=(0, 0)
        )
        
        self.create_control_instructions()  # Create control instructions in English
        
    def create_control_instructions(self):
        """Creates text with control instructions"""
        self.instructions = Text(
            text=dedent('''
            <white>CONTROLS:
            WASD</white> - 
                <yellow>Move horizontally
            <white>Space/Shift</white> - 
                <yellow>Move up/down
            <white>Mouse</white> - 
                <yellow>Look around
            <white>Alt</white> - 
                <yellow>Toggle cursor lock
            <white>Q</white> - 
                <yellow>Quit application
            ''').strip(),
            position=(-0.75, 0.45),
            scale=0.75,
            color=color.white,
            background=True,
            background_color=color.rgba(0, 0, 0, 0.7),
        )
        self.instruction_visible = True
        
        self.toggle_instructions_timer = 0  # Add key for hiding/showing instructions

    def update_position_info(self):
        # Используем фиксированную ширину для стабильного размера текста
        pos_x, pos_y, pos_z = np.round(self.player.position, 3)
        rot_z = 0
        rot_x = self.player.camera_pivot.rotation_x
        rot_y = self.player.rotation_y
        
        self.position_info.text  =  f"Position: {pos_x:>7.3f}, {pos_y:>7.3f}, {pos_z:>7.3f}"
        self.position_info.text += f"\nRotation: {rot_x:>7.3f}, {rot_y:>7.3f}, {rot_z:>7.3f}"
        self.position_info.background = True


        
    def update(self, dt):
        """Updates additional parameters not included in FirstPersonController"""
        self.player.y += (held_keys['space'] - held_keys['shift']) * self.player.speed * dt  # Vertical movement (up/down)
        
        # Show/hide instructions
        if held_keys['h'] and time.time() - self.toggle_instructions_timer > 0.3:
            self.instruction_visible = not self.instruction_visible
            self.instructions.enabled = self.instruction_visible
            self.toggle_instructions_timer = time.time()
            
        if held_keys['alt'] and not hasattr(self, 'alt_timer'):  # Handle Alt press for unlocking/locking cursor
            self.alt_timer = time.time()
        
        if not held_keys['alt'] and hasattr(self, 'alt_timer'):
            if time.time() - self.alt_timer < 0.5:  # Check that it was a short press
                self.toggle_cursor_lock()  # If Alt was pressed and released
            delattr(self, 'alt_timer')
            
        self.update_position_info()
    
    def toggle_cursor_lock(self):
        """Toggles cursor lock"""
        self.cursor_locked = not self.cursor_locked
        
        if self.cursor_locked:
            mouse.locked = True
            self.player.enabled = True
            self.cursor_status.text = "Cursor locked [Alt to unlock]"
        else:
            mouse.locked = False
            self.player.enabled = False  # Disable the controller so the camera doesn't rotate
            self.cursor_status.text = "Cursor unlocked [Alt to lock]"
    
    def input_handler(self, key):
        """Input handler for program closing and speed control"""
        if key == 'q':
            application.quit()

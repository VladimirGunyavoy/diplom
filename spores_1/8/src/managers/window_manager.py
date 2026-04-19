from ursina import *
from .color_manager import ColorManager

# Class for window setup and configuration
class WindowManager:
    def __init__(self, color_manager=None):
        # Используем переданный ColorManager или создаем новый
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager
        
        window.title = '3D Simulation'  # Window setup
        # window.borderless = True # <<< Закомментировано для теста
        # window.fullscreen = True
        
        window.exit_button.visible = True
        # window.fps_counter.enabled = True
        window.color = self.color_manager.get_color('scene', 'window_background')
        window.position = (0, 0)  # <<< Закомментировано для теста
        # window.size = (3000, 1700)  # <<< Закомментировано для теста

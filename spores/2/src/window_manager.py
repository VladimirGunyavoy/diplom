from ursina import *
# Class for window setup and configuration
class WindowManager:
    def __init__(self):
        window.title = '3D Simulation'  # Window setup
        # window.borderless = True # <<< Закомментировано для теста
        # window.fullscreen = True
        
        window.exit_button.visible = True
        # window.fps_counter.enabled = True
        window.color = color.rgb(0.08, 0.1, 0.2)  # Светлее синий фон
        window.position = (0, 0)  # <<< Закомментировано для теста
        # window.size = (3000, 1700)  # <<< Закомментировано для теста

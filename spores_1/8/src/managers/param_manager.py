from ursina import *
from .color_manager import ColorManager
from src.ui.ui_manager import UIManager

class ParamManager():
    def __init__(self, init_value, show=False, color_manager=None, ui_manager=None):
        self.param = init_value

        # Используем переданный ColorManager или создаем новый
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager

        # Используем переданный UIManager или создаем новый
        if ui_manager is None:
            ui_manager = UIManager(color_manager)
        self.ui_manager = ui_manager

        # UI элемент теперь создается в UI_setup.py
        self.show = show
    
    def update(self):
        text = f'param value: {round(self.param, 3)}'
        self.ui_manager.update_text('param_value', text)

    def input_handler(self, key):
        if key == 'z':
            self.param -= 0.1
        if key == 'x':
            self.param += 0.1
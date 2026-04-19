from ursina import *
from .color_manager import ColorManager
from ..visual.ui_manager import UIManager
from typing import Optional

class ParamManager():
    def __init__(self, init_value: float, 
                 show: bool = False, 
                 color_manager: Optional[ColorManager] = None, 
                 ui_manager: Optional[UIManager] = None):
        
        self.param: float = init_value
        self.step: float = 0.05  # Шаг изменения параметра

        # Используем переданный ColorManager или создаем новый
        self.color_manager: ColorManager = color_manager if color_manager is not None else ColorManager()
        
        # Используем переданный UIManager или создаем новый
        self.ui_manager: UIManager = ui_manager if ui_manager is not None else UIManager(self.color_manager)

        # UI элемент теперь создается в UI_setup.py
        self.show: bool = show
    
    def get(self) -> float:
        return self.param
        
    def set(self, value: float) -> None:
        self.param = value
        
    def increase(self) -> None:
        """Увеличивает параметр на заданный шаг."""
        self.set(self.get() + self.step)

    def decrease(self) -> None:
        """Уменьшает параметр на заданный шаг."""
        self.set(self.get() - self.step)

    def update(self) -> None:
        text = f'param value: {round(self.param, 3)}'
        self.ui_manager.update_text('param_value', text)

    def input_handler(self, key: str) -> None:
        if key == 'z':
            self.param -= 0.1
        if key == 'x':
            self.param += 0.1
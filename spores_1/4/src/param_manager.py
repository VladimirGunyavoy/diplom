from ursina import *
from .color_manager import ColorManager

class ParamManager():
    def __init__(self, init_value, show=False, color_manager=None):
        self.param = init_value

        # Используем переданный ColorManager или создаем новый
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager

        self.instruction_text = Text(text='', 
                                     position=(0, 0), 
                                     scale=1,
                                     color=self.color_manager.get_color('ui', 'text_primary'),
                                     background=True,
                                     background_color=self.color_manager.get_color('ui', 'background_solid'))

        if show:
            self.instruction_text.show()
        else:
            self.instruction_text.hide()
    
    def update(self):
        self.instruction_text.text = f'param value: {round(self.param, 3)}'
        self.instruction_text.position = (self.param, -0.0)
        self.instruction_text.background = True

    def input_handler(self, key):
        if key == 'z':
            self.param -= 0.1
        if key == 'x':
            self.param += 0.1
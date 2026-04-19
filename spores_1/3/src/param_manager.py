from ursina import *

class ParamManager():
    def __init__(self, init_value, show=False):
        self.param = init_value

        self.instruction_text = Text(text='', 
                                     position=(0, 0), 
                                     scale=1,
                                     color=color.white,
                                     background=True,
                                     background_color=color.black)

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
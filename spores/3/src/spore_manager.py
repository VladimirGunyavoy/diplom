from ursina import *
import numpy as np

from src.spore import Spore
from src.pendulum import PendulumSystem

class SporeManager():
    def __init__(self, pendulum: PendulumSystem, zoom_manager, settings_param):
        self.objects = []   
        self.pendulum = pendulum
        self.zoom_manager = zoom_manager
        self.settings_param = settings_param
        self.ghost_spore = None
        self.links = []

        self.instruction_text = Text(text='F - new spore', 
                                     position=(-0.75, -0.0), 
                                     scale=0.75,
                                     color=color.white,
                                     background=True,
                                     background_color=color.black)

    def add_spore(self, spore: Spore):
        self.objects.append(spore)
        self.update_ghost_spore()

    def update_ghost_spore(self):
        if not self.objects:
            return
        
        predicted_pos = self.objects[-1].predict_next_position()

        if self.ghost_spore is None:
            self.ghost_spore = self.objects[-1].clone()
            self.ghost_spore.color = color.rgba(255, 255, 255, 120)
            self.zoom_manager.register_object(self.ghost_spore, 'ghost')

        self.ghost_spore.real_position = predicted_pos
        self.ghost_spore.apply_transform(
            self.zoom_manager.a_transformation,
            self.zoom_manager.b_translation,
            spores_scale=self.zoom_manager.spores_scale
        )

    def generate_new_spore(self):
        if not self.objects:
            return
        new_spore = self.objects[-1].step()
        self.add_spore(new_spore)
        self.zoom_manager.register_object(new_spore)
        self.update_ghost_spore()
        return new_spore
    
    def input_handler(self, key):
        if key == 'f':
            self.generate_new_spore()


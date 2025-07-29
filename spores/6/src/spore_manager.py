from ursina import *
import numpy as np

from src.spore import Spore
from src.pendulum import PendulumSystem
from src.link import Link
from src.color_manager import ColorManager
from src.zoom_manager import ZoomManager

class SporeManager():
    def __init__(self, 
                 pendulum: PendulumSystem, 
                 zoom_manager: ZoomManager, 
                 settings_param, 
                 color_manager=None):
        self.objects = []   
        self.pendulum = pendulum
        self.zoom_manager = zoom_manager
        self.settings_param = settings_param
        self.ghost_spore = None
        self.ghost_spores = None
        self.ghost_link = None
        self.links = []

        # Используем переданный ColorManager или создаем новый
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager

        self.instruction_text = Text(text='F - new spore', 
                                     position=(-0.75, -0.0), 
                                     scale=0.75,
                                     color=self.color_manager.get_color('ui', 'text_primary'),
                                     background=True,
                                     background_color=self.color_manager.get_color('ui', 'background_solid'))

    def add_spore(self, spore: Spore):
        self.objects.append(spore)
        self.update_ghost_spore()
        self.sample_ghost_spores(5)

    def update_ghost_spore(self):
        if not self.objects:
            return
        
        predicted_pos = self.objects[-1].evolve_3d()

        if self.ghost_spore is None:
            self.ghost_spore = self.objects[-1].clone()
            self.ghost_spore.color = self.color_manager.get_color('spore', 'ghost')
            self.zoom_manager.register_object(self.ghost_spore, 'ghost_spore')
            

        self.ghost_spore.real_position = predicted_pos
        self.ghost_spore.apply_transform(
            self.zoom_manager.a_transformation,
            self.zoom_manager.b_translation,
            spores_scale=self.zoom_manager.spores_scale
        )

        # Создаем/обновляем призрачный линк
        if self.ghost_link is None:
            self.ghost_link = Link(self.objects[-1], 
                                   self.ghost_spore, 
                                   zoom_manager=self.zoom_manager,
                                   color_manager=self.color_manager)
            self.ghost_link.color = self.color_manager.get_color('link', 'ghost')
            self.zoom_manager.register_object(self.ghost_link, 'ghost_link')
            self.ghost_link.update_geometry()
        else:
            # Обновляем позиции призрачного линка
            self.ghost_link.parent_spore = self.objects[-1]
            self.ghost_link.child_spore = self.ghost_spore
            self.ghost_link.update_geometry()
        
        # Применяем трансформацию
        self.ghost_link.apply_transform(
            self.zoom_manager.a_transformation,
            self.zoom_manager.b_translation,
            spores_scale=self.zoom_manager.spores_scale
        )

    def create_ghost_spores(self, N):
        self.ghost_spores = [self.objects[-1].clone() for _ in range(N)]
        for i, spore in enumerate(self.ghost_spores):
            spore.color = self.color_manager.get_color('spore', 'ghost')
            self.zoom_manager.register_object(spore, f'ghost_spore_{i}')
            spore.name = f'ghost_spore_{i}'

    def sample_ghost_spores(self, N):
        controls = self.pendulum.sample_controls(N)
        states = []
        for i in range(N):
            states.append(self.objects[-1].evolve_3d(control=controls[i]))

        if self.ghost_spores is None:
            self.create_ghost_spores(N)

        for i, spore in enumerate(self.ghost_spores):
            spore.real_position = states[i]
            spore.apply_transform(
                self.zoom_manager.a_transformation,
                self.zoom_manager.b_translation,
                spores_scale=self.zoom_manager.spores_scale
            )



    
    def generate_new_spore(self):
        if not self.objects:
            return
            
        parent_spore = self.objects[-1]
        new_spore = parent_spore.step()
        self.add_spore(new_spore)
        self.zoom_manager.register_object(new_spore)

        # Create a link
        new_link = Link(parent_spore, 
                        new_spore, 
                        zoom_manager=self.zoom_manager,
                        color_manager=self.color_manager)
        self.links.append(new_link)
        self.zoom_manager.register_object(new_link, f"link_{len(self.links)}")

        # After creating a new spore, we need to update the ghost's position
        self.update_ghost_spore()
        return new_spore
    
    def input_handler(self, key):
        if key == 'f':
            self.generate_new_spore()


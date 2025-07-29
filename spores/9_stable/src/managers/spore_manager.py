from ursina import *
import numpy as np

from src.core.spore import Spore
from src.core.pendulum import PendulumSystem
from src.scene.link import Link
from src.managers.color_manager import ColorManager
from .zoom_manager import ZoomManager

class SporeManager():
    def __init__(self, pendulum, zoom_manager, settings_param, 
                 color_manager, angel_manager=None, config=None):
        self.pendulum = pendulum
        self.zoom_manager = zoom_manager
        self.settings_param = settings_param
        self.color_manager = color_manager
        self.angel_manager = angel_manager
        self.config = config

        self.objects = []
        self.trajectories = {}
        self.spore_count = 0
        self.ghost_spores = None
        self.ghost_link = None # Для связи с призраком
        self.links = []
        self._next_spore_id = 0

    def clear(self):
        """Удаляет все споры, связи и другие объекты со сцены."""
        for spore in self.objects:
            destroy(spore)
        for link in self.links:
            destroy(link)
        if self.ghost_link:
            destroy(self.ghost_link)
        for spore in (self.ghost_spores or []):
            destroy(spore)

        self.objects = []
        self.links = []
        self.ghost_spores = []
        self.ghost_link = None
        self._next_spore_id = 0
        
        if self.angel_manager:
            self.angel_manager.clear_all()

    def add_spore(self, spore):
        """Добавляет спору в список управления."""
        if not isinstance(spore.id, int): # Присваиваем ID, если его еще нет
            spore.id = self._next_spore_id
            self._next_spore_id += 1
        
        self.objects.append(spore)

        if self.angel_manager:
            self.angel_manager.on_spore_created(spore)

        self.sample_ghost_spores(N=5) # Обновляем сетку призраков
        self.update_ghost_link()      # Обновляем призрачную связь

    def input_handler(self, key):
        """Обрабатывает ввод для создания новых спор."""
        if held_keys['f']:
            self.generate_new_spore()

    def generate_new_spore(self):
        """Создает новую спору на основе последней."""
        if not self.objects:
            return

        parent_spore = self.objects[-1]
        new_spore = parent_spore.step()
        # ID будет присвоен в add_spore
        
        self.add_spore(new_spore)
        self.zoom_manager.register_object(new_spore, f"spore_{new_spore.id}")

        # Create a link
        new_link = Link(parent_spore,
                        new_spore,
                        color_manager=self.color_manager,
                        zoom_manager=self.zoom_manager,
                        config=self.config)
        self.links.append(new_link)
        self.zoom_manager.register_object(new_link, f"link_{len(self.links)}")

        # --- ИСПРАВЛЕНИЕ (финальная, простая версия): ---
        # Вызываем полный цикл обновления, как это происходит при зуме.
        # Это гарантирует, что все объекты, включая новый линк и его родителей,
        # будут в согласованном состоянии.
        self.zoom_manager.update_transform()

        return new_spore

    def get_last_active_spore(self):
        """Получить последнюю активную спору (не goal)"""
        for spore in reversed(self.objects):
            if not (hasattr(spore, 'is_goal') and spore.is_goal):
                return spore
        return None

    def update_ghost_link(self):
        """Создает и обновляет связь от последней споры к призраку (нулевое управление)."""
        last_spore = self.get_last_active_spore()
        if not last_spore:
            return

        # Получаем одну спору-призрака с нулевым управлением
        control = np.array([0.0])
        state_2d = last_spore.logic.simulate_controls(control)[0]
        
        # Создаем или используем существующую спору-призрак для связи
        if not self.ghost_link:
            ghost_spore = last_spore.clone()
            ghost_spore.is_ghost = True
            ghost_spore.id = "ghost_link_spore"
            ghost_spore.color = self.color_manager.get_color('spore', 'ghost')
            self.zoom_manager.register_object(ghost_spore, ghost_spore.id)
            
            self.ghost_link = Link(last_spore, ghost_spore, 
                                   color_manager=self.color_manager, 
                                   zoom_manager=self.zoom_manager,
                                   config=self.config)
            self.ghost_link.color = self.color_manager.get_color('link', 'ghost')
            self.zoom_manager.register_object(self.ghost_link, "ghost_link")
        
        # Обновляем позицию призрака
        ghost_spore = self.ghost_link.child_spore
        ghost_spore.real_position = np.array([state_2d[0], ghost_spore.y, state_2d[1]])
        ghost_spore.sync_logic_with_position()
        ghost_spore.apply_transform(self.zoom_manager.a_transformation,
                                    self.zoom_manager.b_translation,
                                    spores_scale=self.zoom_manager.spores_scale)

        # Обновляем связь
        self.ghost_link.parent_spore = last_spore
        self.ghost_link.update_geometry()
        
        # Уведомляем AngelManager
        if self.angel_manager:
            self.angel_manager.update_ghost_link_angel(last_spore, ghost_spore)

    def create_ghost_spores(self, N):
        """Создает N призрачных спор."""
        last_spore = self.get_last_active_spore()
        if not last_spore:
            return

        # Перед созданием новых очищаем старые, если они есть
        for spore in (self.ghost_spores or []):
            destroy(spore)

        self.ghost_spores = [last_spore.clone() for _ in range(N)]
        for i, spore in enumerate(self.ghost_spores):
            spore.is_ghost = True
            spore.id = f"ghost_{i}"
            spore.color = self.color_manager.get_color('spore', 'ghost')
            self.zoom_manager.register_object(spore, spore.id)

        if self.angel_manager:
            self.angel_manager.create_ghost_visuals(self.ghost_spores)

    def sample_ghost_spores(self, N=5):
        """Сэмплирует и обновляет позиции призрачных спор (только сетка)."""
        if self.ghost_spores is None:
            self.create_ghost_spores(N)
        
        if not self.ghost_spores:
            return

        last_spore = self.get_last_active_spore()
        if not last_spore:
            return

        # Получаем сэмплы из логики (исключая нулевое управление, чтобы не дублировать)
        controls = last_spore.logic.sample_controls(N, method='mesh')
        states = last_spore.logic.simulate_controls(controls)

        for i, spore in enumerate(self.ghost_spores):
            pos_2d = states[i]
            spore.real_position = np.array([pos_2d[0], spore.y, pos_2d[1]])
            spore.sync_logic_with_position()
            spore.apply_transform(self.zoom_manager.a_transformation,
                                  self.zoom_manager.b_translation,
                                  spores_scale=self.zoom_manager.spores_scale)
        
        # Уведомляем AngelManager о необходимости обновить сетку призрачных ангелов
        if self.angel_manager:
            self.angel_manager.update_ghost_angel_visuals(self.ghost_spores)




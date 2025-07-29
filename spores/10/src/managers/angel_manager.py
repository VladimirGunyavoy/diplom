from ursina import *
from src.visual.link import Link
from src.visual.pillar import Pillar
import numpy as np

class AngelManager:
    # Коэффициент для отступа ангелов от поверхности коста
    ANGEL_HEIGHT_OFFSET_RATIO = 0.01
    
    def __init__(self, color_manager=None, zoom_manager=None, config=None):
        self.color_manager = color_manager
        self.zoom_manager = zoom_manager
        self.cost_function = None # Будет установлено извне
        self.config = config
        
        self.angels = []
        self.pillars = []
        self.links = []

        self.ghost_angels = []
        self.ghost_pillars = []
        self.ghost_link_angel = None # Одна связь для призрака с нулевым управлением

    def clear_all(self):
        """Удаляет все сущности, управляемые этим менеджером."""
        self.clear_ghosts()
        for e in self.angels + self.pillars + self.links:
            destroy(e)
        self.angels, self.pillars, self.links = [], [], []

    def clear_ghosts(self):
        """Удаляет все призрачные сущности."""
        if self.ghost_link_angel:
            # destroy(self.ghost_link_angel.parent_spore) # Родитель - реальный ангел, его удалять не нужно
            destroy(self.ghost_link_angel.child_spore) # Удаляем ангела-потомка
            destroy(self.ghost_link_angel)
            self.ghost_link_angel = None
        for e in self.ghost_angels + self.ghost_pillars:
            destroy(e)
        self.ghost_angels, self.ghost_pillars = [], []

    def on_spore_created(self, spore):
        """
        Вызывается SporeManager-ом при создании ТОЛЬКО НАСТОЯЩЕЙ споры.
        Создает столб, ангела и связь.
        """
        angel_config = self.config.get('angel', {})
        
        # Эта функция теперь не должна вызываться для призраков
        if (hasattr(spore, 'is_ghost') and spore.is_ghost) or not self.cost_function:
            return

        # --- Код ниже выполняется только для НАСТОЯЩИХ спор ---

        spore_2d_pos = spore.calc_2d_pos()
        display_height = self.cost_function.get_cost(spore_2d_pos)
        y_offset = display_height * self.ANGEL_HEIGHT_OFFSET_RATIO

        # Создаем столб, если включено в конфиге
        if angel_config.get('show_pillars', True):
            pillar_width = angel_config['pillar_width']
            pillar = Pillar(
                model='cube',
                color=self.color_manager.get_color('angel', 'pillar'),
            )
            pillar.real_position = spore.real_position + np.array([0, (display_height + y_offset) / 2, 0])
            pillar.real_scale = np.array([pillar_width, display_height + y_offset, pillar_width])
            self.pillars.append(pillar)
            if self.zoom_manager:
                self.zoom_manager.register_object(pillar, f"pillar_{spore.id}")

        # Создаем ангела, если включено в конфиге
        angel = None
        if angel_config.get('show_angels', True):
            angel_real_position = spore.real_position + np.array([0, display_height + y_offset, 0])
            angel = spore.clone(new_position=angel_real_position)
            if spore.is_goal:
                angel.color = self.color_manager.get_color('spore', 'goal')
            else:
                angel.color = self.color_manager.get_color('angel', 'default')
            self.angels.append(angel)
            if self.zoom_manager:
                self.zoom_manager.register_object(angel, f"angel_{spore.id}")

        # Создание связи, если включено и ангелы существуют
        if angel and angel_config.get('show_links', True):
            last_non_goal_angel = None
            for a in reversed(self.angels[:-1]):
                if not a.is_goal and not (hasattr(a, 'is_ghost') and a.is_ghost):
                    last_non_goal_angel = a
                    break
            if not angel.is_goal and last_non_goal_angel:
                new_link = Link(parent_spore=last_non_goal_angel, child_spore=angel, color_manager=self.color_manager, zoom_manager=self.zoom_manager, config=self.config, link_type='angel')
                new_link.color = self.color_manager.get_color('angel', 'link')
                self.links.append(new_link)
                if self.zoom_manager:
                    self.zoom_manager.register_object(new_link, f"angel_link_{len(self.links)}")
                    new_link.update_geometry()
                    new_link.apply_transform(self.zoom_manager.a_transformation, self.zoom_manager.b_translation, spores_scale=self.zoom_manager.spores_scale)

    def update_ghost_link_angel(self, parent_spore, ghost_spore_child):
        """Создает/обновляет ангельскую связь для призрака с нулевым управлением."""
        if not self.cost_function:
            return

        # Находим родительского ангела (последнего настоящего)
        parent_angel = self.angels[-1] if self.angels else None
        if not parent_angel:
            return

        if not self.ghost_link_angel:
            # Создаем ангела-ребенка
            child_angel = ghost_spore_child.clone()
            child_angel.color = self.color_manager.get_color('angel', 'ghost')
            self.zoom_manager.register_object(child_angel, "ghost_link_angel_child")

            # Создаем связь
            self.ghost_link_angel = Link(parent_spore=parent_angel, child_spore=child_angel, color_manager=self.color_manager, zoom_manager=self.zoom_manager, config=self.config, link_type='angel')
            self.ghost_link_angel.color = self.color_manager.get_color('angel', 'ghost_link')
            self.zoom_manager.register_object(self.ghost_link_angel, "ghost_link_angel")
        
        # Обновляем позицию ангела-ребенка
        child_angel = self.ghost_link_angel.child_spore
        spore_2d_pos = ghost_spore_child.calc_2d_pos()
        display_height = self.cost_function.get_cost(spore_2d_pos)
        y_offset = display_height * self.ANGEL_HEIGHT_OFFSET_RATIO
        child_angel.real_position = ghost_spore_child.real_position + np.array([0, display_height + y_offset, 0])
        
        # Обновляем саму связь
        self.ghost_link_angel.parent_spore = parent_angel
        self.ghost_link_angel.update_geometry()
        self.ghost_link_angel.apply_transform(self.zoom_manager.a_transformation, self.zoom_manager.b_translation, spores_scale=self.zoom_manager.spores_scale)

    def create_ghost_visuals(self, ghost_spores):
        """Создает ангелов и столбы для сетки призраков (БЕЗ СВЯЗЕЙ)."""
        if not self.cost_function or not ghost_spores:
            return

        angel_config = self.config.get('angel', {})

        # Очищаем старые объекты (кроме ghost_link_angel)
        for e in self.ghost_angels + self.ghost_pillars:
            destroy(e)
        self.ghost_angels, self.ghost_pillars = [], []

        pillar_width = angel_config.get('ghost_pillar_width', angel_config['pillar_width'])

        for i, ghost_spore in enumerate(ghost_spores):
            # Создаем ангела
            if angel_config.get('show_angels', True):
                angel = ghost_spore.clone()
                angel.color = self.color_manager.get_color('angel', 'ghost')
                self.ghost_angels.append(angel)
                self.zoom_manager.register_object(angel, f"ghost_angel_{i}")

            # Создаем столб
            if angel_config.get('show_pillars', True):
                pillar = Pillar(model='cube', color=self.color_manager.get_color('angel', 'ghost_pillar'))
                self.ghost_pillars.append(pillar)
                self.zoom_manager.register_object(pillar, f"ghost_pillar_{i}")
        
        # Обновляем их визуал
        self.update_ghost_angel_visuals(ghost_spores)

    def update_ghost_angel_visuals(self, ghost_spores):
        """Обновляет позиции и размеры существующих призрачных ангелов и столбов."""
        if not self.cost_function or len(ghost_spores) == 0:
            return

        angel_config = self.config.get('angel', {})
        pillar_width = angel_config.get('ghost_pillar_width', angel_config['pillar_width'])

        for i, ghost_spore in enumerate(ghost_spores):
            spore_2d_pos = ghost_spore.calc_2d_pos()
            display_height = self.cost_function.get_cost(spore_2d_pos)
            y_offset = display_height * self.ANGEL_HEIGHT_OFFSET_RATIO

            # Обновляем ангела
            if i < len(self.ghost_angels) and angel_config.get('show_angels', True):
                angel = self.ghost_angels[i]
                angel.real_position = ghost_spore.real_position + np.array([0, display_height + y_offset, 0])
                angel.apply_transform(self.zoom_manager.a_transformation, self.zoom_manager.b_translation, spores_scale=self.zoom_manager.spores_scale)

            # Обновляем столб
            if i < len(self.ghost_pillars) and angel_config.get('show_pillars', True):
                pillar = self.ghost_pillars[i]
                pillar.real_position = ghost_spore.real_position + np.array([0, (display_height + y_offset) / 2, 0])
                pillar.real_scale = np.array([pillar_width, display_height + y_offset, pillar_width])
                pillar.apply_transform(self.zoom_manager.a_transformation, self.zoom_manager.b_translation, spores_scale=self.zoom_manager.spores_scale) 
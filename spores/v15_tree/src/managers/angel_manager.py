from ursina import destroy
from ..visual.link import Link
from ..visual.pillar import Pillar
import numpy as np
from typing import Optional, List, Dict

# Предварительное объявление классов для аннотаций типов
from ..managers.color_manager import ColorManager
from ..managers.zoom_manager import ZoomManager
from ..logic.cost_function import CostFunction
from ..core.spore import Spore

class AngelManager:
    # Коэффициент для отступа ангелов от поверхности коста
    ANGEL_HEIGHT_OFFSET_RATIO: float = 0.01
    
    def __init__(self, 
                 color_manager: Optional[ColorManager] = None, 
                 zoom_manager: Optional[ZoomManager] = None, 
                 config: Optional[Dict] = None):
        
        self.color_manager: Optional[ColorManager] = color_manager
        self.zoom_manager: Optional[ZoomManager] = zoom_manager
        self.cost_function: Optional[CostFunction] = None # Будет установлено извне
        self.config: Dict = config if config is not None else {}
        
        self.angels: List[Spore] = []
        self.pillars: List[Pillar] = []
        self.links: List[Link] = []

        self.ghost_link_angel: Optional[Link] = None # Одна связь для призрака с нулевым управлением
        
        # Состояние видимости ангелов
        self.angels_visible: bool = config.get('angel', {}).get('show_angels', False)
        self.pillars_visible: bool = config.get('angel', {}).get('show_pillars', False)

    def clear_all(self) -> None:
        """Удаляет все сущности, управляемые этим менеджером."""
        self.clear_ghosts()
        for e in self.angels + self.pillars + self.links:
            if hasattr(e, 'is_ghost') and e.is_ghost:
                continue
            destroy(e)
        self.angels, self.pillars, self.links = [], [], []

    def clear_ghosts(self) -> None:
        """Удаляет все призрачные сущности."""
        if self.ghost_link_angel:
            # destroy(self.ghost_link_angel.parent_spore) # Родитель - реальный ангел, его удалять не нужно
            destroy(self.ghost_link_angel.child_spore) # Удаляем ангела-потомка
            destroy(self.ghost_link_angel)
            self.ghost_link_angel = None

    def on_spore_created(self, spore: Spore) -> None:
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
            pillar.enabled = self.pillars_visible  # Устанавливаем видимость согласно флагу
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
            angel.enabled = self.angels_visible  # Устанавливаем видимость согласно флагу
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
            if not angel.is_goal and last_non_goal_angel and self.zoom_manager:
                new_link = Link(parent_spore=last_non_goal_angel, child_spore=angel, color_manager=self.color_manager, zoom_manager=self.zoom_manager, config=self.config, link_type='angel')
                new_link.color = self.color_manager.get_color('angel', 'link')
                new_link.enabled = self.angels_visible  # Устанавливаем видимость согласно флагу
                self.links.append(new_link)
                link_id = self.zoom_manager.get_unique_link_id()
                self.zoom_manager.register_object(new_link, link_id)
                new_link._zoom_manager_key = link_id  # Сохраняем для удаления
                new_link.update_geometry()
                new_link.apply_transform(self.zoom_manager.a_transformation, self.zoom_manager.b_translation, spores_scale=self.zoom_manager.spores_scale)

    def update_ghost_link_angel(self, parent_spore: Spore, ghost_spore_child: Spore) -> None:
        """Создает/обновляет ангельскую связь для призрака с нулевым управлением."""
        if not self.cost_function or not self.zoom_manager:
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
            self.ghost_link_angel.enabled = self.angels_visible  # Устанавливаем видимость согласно флагу
            child_angel.enabled = self.angels_visible  # И для ангела-ребенка тоже
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
    
    def toggle_angels(self) -> None:
        """Переключает видимость всех ангелов, столбов и связей."""
        self.angels_visible = not self.angels_visible
        self.pillars_visible = not self.pillars_visible
        
        # Переключаем видимость ангелов
        for angel in self.angels:
            angel.enabled = self.angels_visible
        
        # Переключаем видимость столбов
        for pillar in self.pillars:
            pillar.enabled = self.pillars_visible
        
        # Переключаем видимость связей
        for link in self.links:
            link.enabled = self.angels_visible  # Связи видны когда видны ангелы
        
        # Переключаем видимость призрачной связи
        if self.ghost_link_angel:
            self.ghost_link_angel.enabled = self.angels_visible
            if hasattr(self.ghost_link_angel, 'child_spore'):
                self.ghost_link_angel.child_spore.enabled = self.angels_visible
        
        status = "включены" if self.angels_visible else "выключены"
        print(f"👼 Ангелы {status}") 
"""
SporeTreeVisual - Графический класс дерева спор для Ursina

Создает в spores/v14_back/src/visual/spore_tree_visual.py
"""

import numpy as np
from typing import List, Dict, Optional, Any
from ursina import Entity, destroy

from ..core.spore import Spore
from ..visual.link import Link
from ..managers.color_manager import ColorManager
from ..managers.zoom_manager import ZoomManager


class SporeTreeVisual:
    """
    ТОЛЬКО визуализация дерева спор.
    
    Читает готовые данные из SporeTree и отображает их в 3D.
    НЕ содержит математики - только graphics.
    """
    
    def __init__(self,
                 color_manager: ColorManager,
                 zoom_manager: ZoomManager,
                 config: dict):
        """
        Инициализация визуализатора.
        
        Args:
            color_manager: Менеджер цветов
            zoom_manager: Менеджер зума  
            config: Конфигурация для создания Spore объектов
        """
        self.color_manager = color_manager
        self.zoom_manager = zoom_manager
        self.config = config
        
        # Ссылка на логическое дерево (устанавливается извне)
        self.tree_logic = None
        
        # 3D объекты
        self.root_spore: Optional[Spore] = None
        self.child_spores: List[Spore] = []
        self.grandchild_spores: List[Spore] = []
        self.child_links: List[Link] = []
        self.grandchild_links: List[Link] = []
        
        # Счетчики ID
        self._spore_counter = 0
        self._link_counter = 0
        
        # Флаг создания
        self.visual_created = False
        
        # Цвета (соответствуют tree_optimization)
        self.child_colors = ['#FF6B6B', '#9B59B6', '#1ABC9C', '#F39C12']
        self.grandchild_colors = [
            '#FF1744', '#9C27B0', '#2196F3', '#4CAF50',
            '#FF9800', '#795548', '#E91E63', '#607D8B'
        ]
        
    def set_tree_logic(self, tree_logic):
        """
        Связывает визуализатор с логическим деревом.
        
        Args:
            tree_logic: Объект SporeTree с готовой логикой
        """
        self.tree_logic = tree_logic
        
    def create_visual(self) -> None:
        """
        Создает 3D визуализацию на основе данных из tree_logic.
        """
        if not self.tree_logic:
            raise ValueError("Сначала установите tree_logic через set_tree_logic()")
            
        if not self.tree_logic._children_created:
            raise ValueError("В tree_logic должны быть созданы дети")
            
        if self.visual_created:
            self.destroy_visual()
            
        # Читаем настройки из конфига
        goal_position = self.config.get('spore', {}).get('goal_position', [0, 0])
        spore_config = self.config.get('spore', {})
        
        # Создаем визуальные объекты
        self._create_root_visual(goal_position, spore_config)
        self._create_children_visual(goal_position, spore_config)
        
        if self.tree_logic._grandchildren_created:
            self._create_grandchildren_visual(goal_position, spore_config)
            
        self.visual_created = True
        
        spore_count = 1 + len(self.child_spores) + len(self.grandchild_spores)
        link_count = len(self.child_links) + len(self.grandchild_links)
        print(f"🎨 Визуализация создана: {spore_count} спор, {link_count} стрелок")
        
    def _create_root_visual(self, goal_position: List[float], spore_config: dict):
        """Создает визуальный корень."""
        root_data = self.tree_logic.root
        
        self.root_spore = Spore(
            pendulum=self.tree_logic.pendulum,  # Берем из логики
            dt=0.1,  # Для корня не важно
            goal_position=goal_position,
            scale=spore_config.get('scale', 0.1),
            position=(root_data['position'][0], 0.0, root_data['position'][1]),
            color_manager=self.color_manager,
            config=spore_config
        )
        
        # Цвет корня как в matplotlib версии
        self.root_spore.color = self.color_manager.get_color('spore', 'default')
        
        # Регистрируем
        
    def _create_children_visual(self, goal_position: List[float], spore_config: dict):
        """Создает визуальных детей на основе данных из tree_logic."""
        self.child_spores.clear()
        self.child_links.clear()
        
        for i, child_data in enumerate(self.tree_logic.children):
            # Создаем спору
            child_spore = Spore(
                pendulum=self.tree_logic.pendulum,
                dt=abs(child_data['dt']),  # Читаем из логики
                goal_position=goal_position,
                scale=spore_config.get('scale', 0.1),
                position=(child_data['position'][0], 0.0, child_data['position'][1]),
                color_manager=self.color_manager,
                config=spore_config
            )
            
            # Цвет ребенка
            child_spore.color = self.color_manager.get_color('spore', 'default')
            
            # Регистрируем
            self.child_spores.append(child_spore)
            
            # Создаем стрелку
            self._create_child_link_visual(i, child_data)
            
    def _create_child_link_visual(self, child_idx: int, child_data: dict):
        """Создает стрелку для ребенка."""
        child_spore = self.child_spores[child_idx]
        
        # Направление стрелки определяется знаком dt из логики
        if child_data['dt'] > 0:  # forward: корень → ребенок
            parent_spore = self.root_spore
            child_link_spore = child_spore
        else:  # backward: ребенок → корень  
            parent_spore = child_spore
            child_link_spore = self.root_spore
            
        # Создаем линк
        link = Link(
            parent_spore=parent_spore,
            child_spore=child_link_spore,
            color_manager=self.color_manager,
            zoom_manager=self.zoom_manager,
            config=self.config
        )
        
        # Цвет стрелки по управлению (читаем из логики)
        if child_data['control'] > 0:  # u_max
            link.color = self.color_manager.get_color('link', 'ghost_max')
        else:  # u_min
            link.color = self.color_manager.get_color('link', 'ghost_min')
            
        # Регистрируем
        link.update_geometry()
        self.child_links.append(link)
        
    def _create_grandchildren_visual(self, goal_position: List[float], spore_config: dict):
        """Создает визуальных внуков на основе данных из tree_logic."""
        self.grandchild_spores.clear()
        self.grandchild_links.clear()
        
        for i, gc_data in enumerate(self.tree_logic.grandchildren):
            # Создаем спору внука
            grandchild_spore = Spore(
                pendulum=self.tree_logic.pendulum,
                dt=abs(gc_data['dt']),  # Читаем из логики
                goal_position=goal_position,
                scale=spore_config.get('scale', 0.1),
                position=(gc_data['position'][0], 0.0, gc_data['position'][1]),
                color_manager=self.color_manager,
                config=spore_config
            )
            
            # Цвет внука
            grandchild_spore.color = self.color_manager.get_color('spore', 'default')
            # Регистрируем
            self.grandchild_spores.append(grandchild_spore)
            
            # Создаем стрелку внука
            self._create_grandchild_link_visual(i, gc_data)
            
    def _create_grandchild_link_visual(self, gc_idx: int, gc_data: dict):
        """Создает стрелку для внука."""
        grandchild_spore = self.grandchild_spores[gc_idx]
        parent_spore = self.child_spores[gc_data['parent_idx']]
        
        # Направление по знаку dt (читаем из логики)
        if gc_data['dt'] > 0:  # forward: родитель → внук
            parent_link = parent_spore
            child_link = grandchild_spore
        else:  # backward: внук → родитель
            parent_link = grandchild_spore  
            child_link = parent_spore
            
        # Создаем линк
        link = Link(
            parent_spore=parent_link,
            child_spore=child_link,
            color_manager=self.color_manager,
            zoom_manager=self.zoom_manager,
            config=self.config
        )
        
        if gc_data['control'] > 0:
            link.color = self.color_manager.get_color('link', 'ghost_max')
        else:
            link.color = self.color_manager.get_color('link', 'ghost_min')

        # Регистрируем
        link.update_geometry()
        self.grandchild_links.append(link)
        
    def sync_with_logic(self) -> None:
        """
        Синхронизирует визуализацию с обновленной логикой.
        
        Вызывается когда tree_logic изменился (новый dt_vector).
        """
        if not self.visual_created or not self.tree_logic:
            return
            
        # Обновляем позиции детей
        self._sync_children_positions()
        
        # Обновляем позиции внуков (если есть)
        if self.tree_logic._grandchildren_created and len(self.grandchild_spores) > 0:
            self._sync_grandchildren_positions()
            
        print("🔄 Визуализация синхронизирована с логикой")
        
    def _sync_children_positions(self):
        """Обновляет позиции детей из логики."""
        for i, child_data in enumerate(self.tree_logic.children):
            if i < len(self.child_spores):
                child_spore = self.child_spores[i]
                # Читаем новую позицию из логики
                new_pos = child_data['position']
                child_spore.real_position = np.array([new_pos[0], 0.0, new_pos[1]])
                
                # Обновляем стрелку
                if i < len(self.child_links):
                    self.child_links[i].update_geometry()
                    
    def _sync_grandchildren_positions(self):
        """Обновляет позиции внуков из логики."""
        for i, gc_data in enumerate(self.tree_logic.grandchildren):
            if i < len(self.grandchild_spores):
                grandchild_spore = self.grandchild_spores[i]
                # Читаем новую позицию из логики  
                new_pos = gc_data['position']
                grandchild_spore.real_position = np.array([new_pos[0], 0.0, new_pos[1]])
                
                # Обновляем стрелку
                if i < len(self.grandchild_links):
                    self.grandchild_links[i].update_geometry()
                    
    def destroy_visual(self) -> None:
        """Уничтожает все визуальные объекты."""
        # Проверяем, что визуализация была создана, чтобы избежать двойного уничтожения
        if not self.visual_created:
            print("⚠️ Попытка уничтожить уже уничтоженную визуализацию - пропускаем")
            return

        # Уничтожаем стрелки внуков
        for link in self.grandchild_links:
            try:
                destroy(link)
            except:
                pass
        self.grandchild_links.clear()
        
        # Уничтожаем внуков
        for spore in self.grandchild_spores:
            try:
                destroy(spore)
            except:
                pass
        self.grandchild_spores.clear()
        
        # Уничтожаем стрелки детей
        for link in self.child_links:
            try:
                destroy(link)
            except:
                pass
        self.child_links.clear()
        
        # Уничтожаем детей
        for spore in self.child_spores:
            try:
                destroy(spore)
            except:
                pass
        self.child_spores.clear()
        
        # Уничтожаем корень
        if self.root_spore:
            try:
                destroy(self.root_spore)
            except:
                pass
            self.root_spore = None
            
        self.visual_created = False
        print("🗑️ Визуализация уничтожена")
        
    def get_visual_info(self) -> Dict[str, Any]:
        """Возвращает информацию о визуализации."""
        return {
            'created': self.visual_created,
            'has_logic': self.tree_logic is not None,
            'spore_count': {
                'root': 1 if self.root_spore else 0,
                'children': len(self.child_spores),
                'grandchildren': len(self.grandchild_spores)
            },
            'link_count': {
                'child_links': len(self.child_links),
                'grandchild_links': len(self.grandchild_links)
            }
        }
        
    def _get_next_spore_id(self) -> int:
        """Генерирует уникальный ID для споры."""
        self._spore_counter += 1
        return self._spore_counter
        
    def _get_next_link_id(self) -> int:
        """Генерирует уникальный ID для стрелки."""
        self._link_counter += 1
        return self._link_counter

    def __del__(self):
        """Деструктор - очищает ресурсы."""
        if hasattr(self, 'visual_created') and self.visual_created:
            self.destroy_visual()
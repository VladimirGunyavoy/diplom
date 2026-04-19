"""
Централизованный менеджер ID для всех объектов в системе спор.

Решает проблему конфликтующих счетчиков в разных модулях:
- manual_spore_manager._spore_counter
- spore_manager._next_spore_id
- spore_tree_visual._spore_counter
- zoom_manager._global_*_counter

Теперь все объекты получают ID из одного источника.
"""

from typing import Dict, Any


class IDManager:
    """
    Централизованный менеджер ID для всех объектов в системе.
    Гарантирует уникальность ID для спор и линков.
    """
    
    def __init__(self):
        # Основные счетчики для объектов симуляции
        self._spore_counter = 0
        self._link_counter = 0
        
        # Дополнительные счетчики для специальных объектов
        self._angel_counter = 0
        self._pillar_counter = 0
        self._ghost_counter = 0
        
    def get_next_spore_id(self) -> int:
        """
        Возвращает уникальный числовой ID для споры.
        
        Returns:
            int: Уникальный ID споры
        """
        self._spore_counter += 1
        return self._spore_counter
    
    def get_next_link_id(self) -> int:
        """
        Возвращает уникальный числовой ID для линка.
        
        Returns:
            int: Уникальный ID линка
        """
        self._link_counter += 1
        return self._link_counter
    
    def get_next_angel_id(self) -> int:
        """Возвращает уникальный ID для ангела."""
        self._angel_counter += 1
        return self._angel_counter
    
    def get_next_pillar_id(self) -> int:
        """Возвращает уникальный ID для столба."""
        self._pillar_counter += 1
        return self._pillar_counter
    
    def get_next_ghost_id(self) -> int:
        """Возвращает уникальный ID для призрачного объекта."""
        self._ghost_counter += 1
        return self._ghost_counter
    
    def reset_counters(self):
        """
        Сбрасывает все счетчики (для полной очистки сцены).
        
        Вызывается при:
        - Очистке всех объектов (clear_all)
        - Перезапуске симуляции
        - Загрузке новой сцены
        """
        self._spore_counter = 0
        self._link_counter = 0
        self._angel_counter = 0
        self._pillar_counter = 0
        self._ghost_counter = 0
        
    def reset_spore_counter(self):
        """Сбрасывает только счетчик спор (частичная очистка)."""
        self._spore_counter = 0
        
    def reset_link_counter(self):
        """Сбрасывает только счетчик линков (частичная очистка)."""
        self._link_counter = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику выданных ID.
        
        Returns:
            dict: Словарь со статистикой по каждому типу объектов
        """
        return {
            'spores_created': self._spore_counter,
            'links_created': self._link_counter,
            'angels_created': self._angel_counter,
            'pillars_created': self._pillar_counter,
            'ghosts_created': self._ghost_counter,
            'total_objects': (self._spore_counter + self._link_counter + 
                            self._angel_counter + self._pillar_counter + 
                            self._ghost_counter)
        }
    
    def get_next_available_spore_id(self) -> int:
        """
        Возвращает следующий ID споры БЕЗ увеличения счетчика.
        Полезно для предварительного планирования.
        
        Returns:
            int: Следующий доступный ID споры
        """
        return self._spore_counter + 1
    
    def get_next_available_link_id(self) -> int:
        """
        Возвращает следующий ID линка БЕЗ увеличения счетчика.
        
        Returns:
            int: Следующий доступный ID линка
        """
        return self._link_counter + 1
    
    def __str__(self) -> str:
        stats = self.get_stats()
        return (f"IDManager(spores: {stats['spores_created']}, "
                f"links: {stats['links_created']}, "
                f"total: {stats['total_objects']})")
    
    def __repr__(self) -> str:
        return self.__str__()

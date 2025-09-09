"""
SporeGraph - Центральное хранилище структуры связей между спорами

Этот класс знает:
- Какая спора с какой соединена
- Направление связи (parent_spore -> child_spore)
- Тип связи (цвет: ghost_max, ghost_min, active, angel)
- Граф может быть реальным или призрачным
"""

from typing import Dict, Optional, Set, List, Tuple, Any
from ..core.spore import Spore
from ..visual.link import Link


class EdgeInfo:
    """Информация о ребре графа между двумя спорами"""

    def __init__(self,
                 parent_spore: Spore,
                 child_spore: Spore,
                 link_type: str = 'default',
                 link_object: Optional[Link] = None):
        self.parent_spore = parent_spore
        self.child_spore = child_spore
        self.link_type = link_type
        self.link_object = link_object

    def get_direction_tuple(self) -> Tuple[str, str]:
        """Возвращает кортеж (parent_id, child_id) для идентификации ребра"""
        return (self.parent_spore.id, self.child_spore.id)

    def __repr__(self):
        return (f"EdgeInfo({self.parent_spore.id} -> {self.child_spore.id}, "
                f"type={self.link_type})")


class SporeGraph:
    """
    Центральный граф связей между спорами.

    Хранит структуру связей отдельно от визуальных объектов Link.
    Позволяет легко копировать структуру между реальным и призрачным графом.
    """

    def __init__(self, graph_type: str = 'real'):
        """
        Args:
            graph_type: 'real' для реального графа, 'ghost' для призрачного
        """
        self.graph_type = graph_type

        # Основное хранилище: от (parent_id, child_id) к EdgeInfo
        self.edges: Dict[Tuple[str, str], EdgeInfo] = {}

        # Индексы для быстрого поиска
        self.nodes: Dict[str, Spore] = {}
        self.outgoing: Dict[str, Set[str]] = {}
        self.incoming: Dict[str, Set[str]] = {}

    def add_spore(self, spore: Spore) -> None:
        """Добавляет спору в граф"""
        if not hasattr(spore, 'id') or spore.id is None:
            raise ValueError(f"Spore должна иметь уникальный id: {spore}")

        self.nodes[spore.id] = spore
        if spore.id not in self.outgoing:
            self.outgoing[spore.id] = set()
        if spore.id not in self.incoming:
            self.incoming[spore.id] = set()

    def add_edge(self,
                 parent_spore: Spore,
                 child_spore: Spore,
                 link_type: str = 'default',
                 link_object: Optional[Link] = None) -> EdgeInfo:
        """
        Добавляет ребро в граф

        Args:
            parent_spore: Спора-родитель (начало стрелки)
            child_spore: Спора-ребенок (конец стрелки)
            link_type: Тип связи (ghost_max, ghost_min, active, angel)
            link_object: Опциональная ссылка на визуальный Link

        Returns:
            EdgeInfo: Информация о добавленном ребре
        """
        # Добавляем споры если их нет
        self.add_spore(parent_spore)
        self.add_spore(child_spore)

        edge_info = EdgeInfo(parent_spore, child_spore, link_type, link_object)
        edge_key = edge_info.get_direction_tuple()

        # Если ребро уже существует, обновляем его
        if edge_key in self.edges:
            print(f"⚠️ SporeGraph: Обновляем существующее ребро {edge_key}")

        self.edges[edge_key] = edge_info

        # Обновляем индексы
        self.outgoing[parent_spore.id].add(child_spore.id)
        self.incoming[child_spore.id].add(parent_spore.id)

        return edge_info

    def remove_edge(self, parent_id: str, child_id: str) -> bool:
        """
        Удаляет ребро из графа

        Returns:
            bool: True если ребро было удалено, False если не найдено
        """
        edge_key = (parent_id, child_id)
        if edge_key not in self.edges:
            return False

        del self.edges[edge_key]

        # Обновляем индексы
        if parent_id in self.outgoing:
            self.outgoing[parent_id].discard(child_id)
        if child_id in self.incoming:
            self.incoming[child_id].discard(parent_id)

        return True

    def get_children(self, parent_id: str) -> List[Spore]:
        """Возвращает всех детей данной споры"""
        child_ids = self.outgoing.get(parent_id, set())
        return [self.nodes[child_id] for child_id in child_ids
                if child_id in self.nodes]

    def get_parents(self, child_id: str) -> List[Spore]:
        """Возвращает всех родителей данной споры"""
        parent_ids = self.incoming.get(child_id, set())
        return [self.nodes[parent_id] for parent_id in parent_ids
                if parent_id in self.nodes]

    def get_edge_info(self, parent_id: str,
                      child_id: str) -> Optional[EdgeInfo]:
        """Возвращает информацию о ребре"""
        return self.edges.get((parent_id, child_id))

    def copy_structure_from(self, other_graph: 'SporeGraph') -> None:
        """
        Копирует структуру связей из другого графа.
        Используется для копирования из призрачного графа в реальный.

        Args:
            other_graph: Граф-источник (обычно призрачный)
        """
        print(f"🔄 Копируем структуру из {other_graph.graph_type} "
              f"графа в {self.graph_type}")
        print(f"   📊 Источник: {len(other_graph.edges)} ребер, "
              f"{len(other_graph.nodes)} узлов")

        for edge_key, edge_info in other_graph.edges.items():
            # Копируем только структуру, link_object не переносим
            self.add_edge(
                edge_info.parent_spore,
                edge_info.child_spore,
                edge_info.link_type,
                link_object=None
            )

        print(f"   ✅ Скопировано: {len(self.edges)} ребер, "
              f"{len(self.nodes)} узлов")

    def clear(self) -> None:
        """Очищает весь граф"""
        self.edges.clear()
        self.nodes.clear()
        self.outgoing.clear()
        self.incoming.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику графа"""
        link_types = {}
        for edge_info in self.edges.values():
            link_types[edge_info.link_type] = (
                link_types.get(edge_info.link_type, 0) + 1)

        return {
            'graph_type': self.graph_type,
            'nodes_count': len(self.nodes),
            'edges_count': len(self.edges),
            'link_types': link_types
        }

    def debug_print(self) -> None:
        """Выводит отладочную информацию о графе"""
        stats = self.get_stats()
        print(f"📊 {self.graph_type.upper()} ГРАФ:")
        print(f"   🔴 Узлов (спор): {stats['nodes_count']}")
        print(f"   🔗 Ребер (связей): {stats['edges_count']}")
        print(f"   🎨 Типы связей: {stats['link_types']}")

        if len(self.edges) <= 10:  # Детали только для небольших графов
            for edge_key, edge_info in self.edges.items():
                print(f"      {edge_info}")

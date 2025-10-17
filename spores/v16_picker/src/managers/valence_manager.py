"""
Valence Manager - Менеджер для анализа валентности спор
========================================================

Управляет анализом и отслеживанием валентных состояний всех спор в графе.
Определяет занятые и свободные валентные места на основе соседей 1 и 2 порядка.
"""

from typing import Dict, Optional, List, Any
import numpy as np
from ..logic.valence import SporeValence


class ValenceManager:
    """
    Менеджер для анализа валентности спор.

    Анализирует граф связей и определяет для каждой споры:
    - Какие валентные слоты заняты (есть соседи)
    - Какие слоты свободны (можно создать новых соседей)
    - Параметры существующих связей (dt, управление)

    Attributes:
        spore_manager: Ссылка на SporeManager для доступа к графу
        valence_cache: Кеш валентности спор
    """

    def __init__(self, spore_manager):
        """
        Инициализация ValenceManager.

        Args:
            spore_manager: SporeManager для доступа к графу связей
        """
        self.spore_manager = spore_manager
        self.valence_cache: Dict[int, SporeValence] = {}

        print("✅ ValenceManager инициализирован")

    def _get_visual_spore_id(self, spore_id: str) -> str:
        """
        Получает визуальный ID споры (как на картинке) - индекс в списке + 1.

        Args:
            spore_id: Внутренний ID споры из графа

        Returns:
            Визуальный ID (строка) или исходный ID если спора не найдена
        """
        try:
            # Получаем объект споры из графа
            if spore_id not in self.spore_manager.graph.nodes:
                return spore_id

            spore = self.spore_manager.graph.nodes[spore_id]

            # Ищем индекс в списке objects (используем тот же алгоритм что и PickerManager)
            real_spores = self.spore_manager.objects
            spore_index = next((i for i, s in enumerate(real_spores) if s is spore), -1)

            if spore_index >= 0:
                return str(spore_index + 1)  # Нумерация от 1 (как на картинке)
            else:
                return spore_id
        except Exception:
            return spore_id

    def analyze_spore_valence(self, spore_id: str, use_cache: bool = True) -> Optional[SporeValence]:
        """
        Анализирует валентность споры на основе ее соседей в графе.

        Args:
            spore_id: ID споры для анализа
            use_cache: Использовать ли кеш валентности

        Returns:
            SporeValence с информацией о валентности или None при ошибке
        """
        # Проверяем кеш
        if use_cache and spore_id in self.valence_cache:
            return self.valence_cache[spore_id]

        # Проверяем что спора существует в графе
        if spore_id not in self.spore_manager.graph.nodes:
            print(f"❌ Спора {spore_id} не найдена в графе")
            return None

        # Создаем пустую валентность
        valence = SporeValence(spore_id=spore_id)

        # Получаем соседей 1-го порядка (дети)
        neighbors_1 = self._get_neighbors_at_distance(spore_id, 1)
        for neighbor in neighbors_1:
            self._occupy_slot_from_neighbor(valence, neighbor, distance=1)

        # Получаем соседей 2-го порядка (внуки)
        neighbors_2 = self._get_neighbors_at_distance(spore_id, 2)
        for neighbor in neighbors_2:
            self._occupy_slot_from_neighbor(valence, neighbor, distance=2)

        # Сохраняем в кеш
        self.valence_cache[spore_id] = valence

        return valence

    def _get_neighbors_at_distance(self, spore_id: str, distance: int) -> List[Dict[str, Any]]:
        """
        Получает соседей споры на заданном расстоянии.

        Использует методы из графа для получения соседей.

        Args:
            spore_id: ID споры
            distance: Расстояние (1 или 2)

        Returns:
            Список информации о соседях
        """
        if distance == 1:
            return self._get_direct_neighbors(spore_id)
        elif distance == 2:
            return self._get_neighbors_at_distance_2(spore_id)
        else:
            return []

    def _get_direct_neighbors(self, spore_id: str) -> List[Dict[str, Any]]:
        """Получает прямых соседей (детей и родителей)."""

        neighbors: List[Dict[str, Any]] = []

        # Исходящие связи (дети)
        children = self.spore_manager.graph.get_children(spore_id)
        for child in children:
            child_id = self.spore_manager.graph._get_spore_id(child)
            edge_info = self.spore_manager.graph.get_edge_info(spore_id, child_id)

            dt_value = self._convert_to_float(self._extract_dt_from_edge(edge_info))
            control_value = self._convert_to_float(self._extract_control_from_edge(edge_info))
            time_direction = self._determine_time_direction(dt_value)

            neighbor_info = {
                'target_spore': child,
                'target_id': child_id,
                'path': [spore_id, child_id],
                'time_direction': 'forward',
                'dt': dt_value,
                'dt_sequence': [dt_value] if dt_value is not None else None,
                'control': control_value,
                'raw_direction': 'outgoing',
            }
            neighbors.append(neighbor_info)

        # Входящие связи (родители) — двигаемся против направления ребра
        parents = self.spore_manager.graph.get_parents(spore_id)
        for parent in parents:
            parent_id = self.spore_manager.graph._get_spore_id(parent)
            edge_info = self.spore_manager.graph.get_edge_info(parent_id, spore_id)

            raw_dt = self._convert_to_float(self._extract_dt_from_edge(edge_info))
            raw_control = self._convert_to_float(self._extract_control_from_edge(edge_info))

            dt_value = -raw_dt if raw_dt is not None else None
            control_value = -raw_control if raw_control is not None else None
            time_direction = self._determine_time_direction(dt_value)

            neighbor_info = {
                'target_spore': parent,
                'target_id': parent_id,
                'path': [spore_id, parent_id],
                'time_direction': 'backward',
                'dt': dt_value,
                'dt_sequence': [dt_value] if dt_value is not None else None,
                'control': control_value,
                'raw_direction': 'incoming',
            }
            neighbors.append(neighbor_info)

        return neighbors

    def _get_neighbors_at_distance_2(self, spore_id: str) -> List[Dict[str, Any]]:
        """Получает соседей на расстоянии 2 (внуки)."""

        neighbors: List[Dict[str, Any]] = []

        direct_neighbors = self._get_direct_neighbors(spore_id)

        for direct_neighbor in direct_neighbors:
            intermediate_id = direct_neighbor['target_id']

            if intermediate_id == spore_id:
                continue

            intermediate_neighbors = self._get_direct_neighbors(intermediate_id)

            for neighbor in intermediate_neighbors:
                target_id = neighbor['target_id']

                if target_id in {spore_id, intermediate_id}:
                    continue

                first_control = direct_neighbor.get('control')
                second_control = neighbor.get('control')

                first_control_type = self._determine_control_type(first_control)
                second_control_type = self._determine_control_type(second_control)

                if not first_control_type or not second_control_type:
                    continue

                if first_control_type == second_control_type:
                    continue

                path = [spore_id, intermediate_id, target_id]

                first_dt = direct_neighbor.get('dt')
                second_dt = neighbor.get('dt')
                dt_sequence = [first_dt, second_dt]
                dt_values = [dt for dt in dt_sequence if dt is not None]
                total_dt = sum(dt_values) if dt_values else None

                first_time_dir = direct_neighbor.get('time_direction')
                second_time_dir = neighbor.get('time_direction')

                neighbor_info = {
                    'target_spore': neighbor.get('target_spore'),
                    'target_id': target_id,
                    'path': path,
                    'intermediate_id': intermediate_id,
                    'intermediate_spore': direct_neighbor.get('target_spore'),
                    'first_time_direction': first_time_dir,
                    'first_control': first_control,
                    'first_control_type': first_control_type,
                    'first_dt': first_dt,
                    'second_time_direction': second_time_dir,
                    'second_control': second_control,
                    'second_control_type': second_control_type,
                    'second_dt': second_dt,
                    'step_time_directions': [first_time_dir, second_time_dir],
                    'control_sequence': [first_control, second_control],
                    'dt_sequence': dt_sequence,
                    'dt': total_dt,
                }
                neighbors.append(neighbor_info)

        return neighbors

    def _extract_dt_from_edge(self, edge_info: Any) -> Optional[float]:
        """
        Извлекает значение dt из информации о связи.

        Args:
            edge_info: Информация о связи из графа

        Returns:
            Значение dt или None
        """
        if not edge_info or not hasattr(edge_info, 'link_object'):
            return None

        link = edge_info.link_object
        if hasattr(link, 'dt_value'):
            dt = link.dt_value
            # Конвертируем numpy в python float
            if isinstance(dt, np.ndarray):
                dt = float(dt.flatten()[0]) if dt.size > 0 else None
            elif dt is not None:
                dt = float(dt)
            return dt

        return None

    def _extract_dt_for_direction(self, edge_info: Any, direction: str) -> Optional[float]:
        """Извлекает dt и приводит знак в соответствии с направлением."""
        raw_dt = self._convert_to_float(self._extract_dt_from_edge(edge_info))
        if raw_dt is None:
            return None

        abs_dt = abs(raw_dt)
        if direction == 'forward':
            return abs_dt
        if direction == 'backward':
            return -abs_dt
        return raw_dt


    def _extract_control_from_edge(self, edge_info: Any) -> Optional[float]:
        """
        Извлекает значение управления из информации о связи.

        Args:
            edge_info: Информация о связи из графа

        Returns:
            Значение управления или None
        """
        if not edge_info or not hasattr(edge_info, 'link_object'):
            return None

        link = edge_info.link_object
        control = None
        if hasattr(link, 'control_value'):
            control = link.control_value
            # Конвертируем numpy в python float
            if isinstance(control, np.ndarray):
                control = float(control.flatten()[0]) if control.size > 0 else None
            elif control is not None:
                control = float(control)

        link_type = getattr(edge_info, 'link_type', '') or ''
        link_type_lower = link_type.lower() if isinstance(link_type, str) else ''
        if 'max' in link_type_lower:
            magnitude = abs(control) if control is not None else 1.0
            control = magnitude
        elif 'min' in link_type_lower:
            magnitude = abs(control) if control is not None else 1.0
            control = -magnitude

        return control

    def _convert_to_float(self, value: Any) -> Optional[float]:
        """Преобразует значение (включая numpy) в float."""
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            value = value.flatten()[0]

        if isinstance(value, (np.floating, np.integer)):
            return float(value)

        if isinstance(value, (float, int)):
            return float(value)

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _determine_time_direction(self, dt_value: Optional[float]) -> str:
        """Определяет направление времени по знаку dt."""
        if dt_value is None:
            return 'forward'
        return 'forward' if dt_value >= 0 else 'backward'

    def _determine_control_type(self, control_value: Optional[float]) -> Optional[str]:
        """Определяет тип управления по знаку control."""
        if control_value is None:
            return None

        if control_value > 0:
            return 'max'
        if control_value < 0:
            return 'min'

        return None

    def _convert_to_float(self, value: Any) -> Optional[float]:
        """Преобразует значение (включая numpy) в float."""
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            value = value.flatten()[0]

        if isinstance(value, (np.floating, np.integer)):
            return float(value)

        if isinstance(value, (float, int)):
            return float(value)

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _determine_time_direction(self, dt_value: Optional[float]) -> str:
        """Определяет направление времени по знаку dt."""
        if dt_value is None:
            return 'forward'
        return 'forward' if dt_value >= 0 else 'backward'

    def _determine_control_type(self, control_value: Optional[float]) -> Optional[str]:
        """Определяет тип управления по знаку control."""
        if control_value is None:
            return None

        if control_value > 0:
            return 'max'
        if control_value < 0:
            return 'min'

        return None

    def _occupy_slot_from_neighbor(self, valence: SporeValence, neighbor: Dict[str, Any], distance: int) -> None:
        """Заполняет слот валентности на основе информации о соседе."""

        if distance == 1:
            time_dir = neighbor.get('time_direction')
            control_type = self._determine_control_type(neighbor.get('control')) or 'max'

            slot = valence.find_slot_by_parameters(
                slot_type='child',
                time_direction=time_dir,
                control_type=control_type
            )

            if slot and not slot.occupied:
                slot.occupied = True
                slot.neighbor_id = neighbor.get('target_id')
                slot.dt_value = neighbor.get('dt')
                slot.dt_sequence = (list(neighbor.get('dt_sequence'))
                                     if neighbor.get('dt_sequence') is not None else None)
                slot.is_fixed = True  # Существующие связи зафиксированы

        elif distance == 2:
            first_time_dir = neighbor.get('first_time_direction')
            first_control_type = neighbor.get('first_control_type')
            second_time_dir = neighbor.get('second_time_direction')
            second_control_type = neighbor.get('second_control_type')

            if not first_control_type or not second_control_type:
                return

            expected_second_type = 'min' if first_control_type == 'max' else 'max'
            if second_control_type != expected_second_type:
                return

            slot = valence.find_slot_by_parameters(
                slot_type='grandchild',
                time_direction=first_time_dir,
                control_type=first_control_type,
                second_time_direction=second_time_dir
            )

            if slot and not slot.occupied:
                slot.occupied = True
                slot.neighbor_id = neighbor.get('target_id')
                slot.dt_value = neighbor.get('dt')
                slot.dt_sequence = (list(neighbor.get('dt_sequence'))
                                     if neighbor.get('dt_sequence') is not None else None)
                slot.is_fixed = True  # Существующие связи зафиксированы

    def clear_cache(self) -> None:
        """Очищает кеш валентности"""
        self.valence_cache.clear()
        print("🧹 Кеш валентности очищен")

    def get_valence_info(self, spore_id: str) -> Optional[SporeValence]:
        """
        Получает информацию о валентности споры.

        Алиас для analyze_spore_valence с использованием кеша.

        Args:
            spore_id: ID споры

        Returns:
            SporeValence или None
        """
        return self.analyze_spore_valence(spore_id, use_cache=True)

    def update_from_graph(self) -> None:
        """
        Обновляет кеш валентности для всех спор в графе.

        Пересчитывает валентность для всех спор.
        """
        print("🔄 Обновление валентности из графа...")

        self.clear_cache()

        # Анализируем все споры в графе
        for spore_id in self.spore_manager.graph.nodes.keys():
            self.analyze_spore_valence(spore_id, use_cache=False)

        print(f"✅ Валентность обновлена для {len(self.valence_cache)} спор")

    def print_valence_report(self, spore_id: str) -> None:
        """
        Выводит подробный отчет о валентности споры.

        Args:
            spore_id: ID споры для отчета
        """
        valence = self.analyze_spore_valence(spore_id)

        if not valence:
            print(f"❌ Не удалось проанализировать валентность споры {spore_id}")
            return

        # Получаем визуальный ID для красивого отображения
        visual_id = self._get_visual_spore_id(spore_id)

        # Выводим основную информацию с визуальными ID
        valence.print_summary(spore_visual_id=visual_id, id_mapper=self._get_visual_spore_id)

        # Выводим данные для оптимизатора
        valence.print_optimizer_data()

    def get_all_valences(self) -> Dict[str, SporeValence]:
        """
        Возвращает валентность всех спор в графе.

        Returns:
            Словарь {spore_id: SporeValence}
        """
        if not self.valence_cache:
            self.update_from_graph()

        return self.valence_cache.copy()

    def print_graph_valence_summary(self) -> None:
        """Выводит сводку по валентности всего графа"""
        print("\n📊 СВОДКА ВАЛЕНТНОСТИ ГРАФА:")

        all_valences = self.get_all_valences()

        if not all_valences:
            print("   📭 Граф пуст")
            return

        # Собираем статистику
        total_spores = len(all_valences)
        total_occupied_children = 0
        total_occupied_grandchildren = 0
        total_free_children = 0
        total_free_grandchildren = 0

        for valence in all_valences.values():
            total_occupied_children += valence.count_occupied_children()
            total_occupied_grandchildren += valence.count_occupied_grandchildren()
            total_free_children += valence.count_free_children()
            total_free_grandchildren += valence.count_free_grandchildren()

        print(f"   🌟 Всего спор: {total_spores}")
        print(f"\n   👶 ДЕТИ:")
        print(f"      Занято слотов: {total_occupied_children}")
        print(f"      Свободно слотов: {total_free_children}")
        print(f"\n   👶👶 ВНУКИ:")
        print(f"      Занято слотов: {total_occupied_grandchildren}")
        print(f"      Свободно слотов: {total_free_grandchildren}")

        # Находим споры с интересной валентностью
        fully_occupied = [vid for vid, v in all_valences.items()
                         if len(v.get_free_slots()) == 0]
        fully_free = [vid for vid, v in all_valences.items()
                     if len(v.get_occupied_slots()) == 0]
        partially_occupied = [vid for vid, v in all_valences.items()
                             if 0 < len(v.get_occupied_slots()) < v.total_slots]

        print(f"\n   🎯 КАТЕГОРИИ СПОР:")
        print(f"      Полностью заняты (8/8): {len(fully_occupied)}")
        print(f"      Полностью свободны (0/8): {len(fully_free)}")
        print(f"      Частично заняты: {len(partially_occupied)}")

        if partially_occupied:
            print(f"\n   🔍 ЧАСТИЧНО ЗАНЯТЫЕ СПОРЫ (интересны для роста дерева):")
            for spore_id in partially_occupied[:5]:  # Показываем первые 5
                v = all_valences[spore_id]
                occupied = len(v.get_occupied_slots())
                visual_id = self._get_visual_spore_id(spore_id)
                print(f"      Спора {visual_id}: {occupied}/8 слотов занято")
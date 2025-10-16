"""
Valence Manager - Менеджер для анализа валентности спор
========================================================

Управляет анализом и отслеживанием валентных состояний всех спор в графе.
Определяет занятые и свободные валентные места на основе соседей 1 и 2 порядка.
"""

from typing import Dict, Optional, List, Any
import numpy as np
from ..logic.valence import ValenceSlot, SporeValence


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
        """
        Получает прямых соседей (детей и родителей).

        Args:
            spore_id: ID споры

        Returns:
            Список соседей с метаданными
        """
        neighbors: List[Dict[str, Any]] = []

        # Исходящие связи (дети)
        children = self.spore_manager.graph.get_children(spore_id)
        for child in children:
            child_id = self.spore_manager.graph._get_spore_id(child)
            edge_info = self.spore_manager.graph.get_edge_info(spore_id, child_id)

            dt_value = self._extract_dt_from_edge(edge_info)
            control_value = self._extract_control_from_edge(edge_info)

            # Направление времени определяется знаком dt:
            # dt > 0 → forward (прямое время)
            # dt < 0 → backward (обратное время)
            time_direction = 'forward' if dt_value and dt_value > 0 else 'backward'

            neighbor_info = {
                'target_spore': child,
                'target_id': child_id,
                'path': [spore_id, child_id],
                'time_direction': time_direction,
                'dt': dt_value,
                'control': control_value,
            }
            neighbors.append(neighbor_info)

        # Входящие связи (родители)
        parents = self.spore_manager.graph.get_parents(spore_id)
        for parent in parents:
            parent_id = self.spore_manager.graph._get_spore_id(parent)
            edge_info = self.spore_manager.graph.get_edge_info(parent_id, spore_id)

            dt_value = self._extract_dt_from_edge(edge_info)
            control_value = self._extract_control_from_edge(edge_info)

            # Направление времени определяется знаком dt:
            # dt > 0 → forward (прямое время)
            # dt < 0 → backward (обратное время)
            time_direction = 'forward' if dt_value and dt_value > 0 else 'backward'

            neighbor_info = {
                'target_spore': parent,
                'target_id': parent_id,
                'path': [spore_id, parent_id],
                'time_direction': time_direction,
                'dt': dt_value,
                'control': control_value,
            }
            neighbors.append(neighbor_info)

        return neighbors

    def _get_neighbors_at_distance_2(self, spore_id: str) -> List[Dict[str, Any]]:
        """
        Получает соседей на расстоянии 2 (внуки).

        Args:
            spore_id: ID споры

        Returns:
            Список внуков с метаданными о пути
        """
        neighbors: List[Dict[str, Any]] = []

        # Получаем прямых соседей
        direct_neighbors = self._get_direct_neighbors(spore_id)

        # Для каждого прямого соседа получаем его соседей
        for direct_neighbor in direct_neighbors:
            intermediate_id = direct_neighbor['target_id']

            # Избегаем циклов
            if intermediate_id == spore_id:
                continue

            # Получаем соседей промежуточной споры
            intermediate_neighbors = self._get_direct_neighbors(intermediate_id)

            for neighbor in intermediate_neighbors:
                target_id = neighbor['target_id']

                # Избегаем возврата к исходной споре и промежуточной
                if target_id in {spore_id, intermediate_id}:
                    continue

                # Формируем информацию о маршруте из 2 шагов
                path = [spore_id, intermediate_id, target_id]

                # Первый шаг: от spore_id к intermediate_id
                first_time_dir = direct_neighbor['time_direction']
                first_control = direct_neighbor['control']
                first_dt = direct_neighbor['dt']

                # Второй шаг: от intermediate_id к target_id
                second_time_dir = neighbor['time_direction']
                second_control = neighbor['control']
                second_dt = neighbor['dt']

                neighbor_info = {
                    'target_spore': neighbor['target_spore'],
                    'target_id': target_id,
                    'path': path,
                    'intermediate_id': intermediate_id,
                    'first_time_direction': first_time_dir,
                    'first_control': first_control,
                    'first_dt': first_dt,
                    'second_time_direction': second_time_dir,
                    'second_control': second_control,
                    'second_dt': second_dt,
                    # Общий dt - сумма двух шагов
                    'dt': (first_dt if first_dt is not None else 0) + (second_dt if second_dt is not None else 0),
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
        if hasattr(link, 'control_value'):
            control = link.control_value
            # Конвертируем numpy в python float
            if isinstance(control, np.ndarray):
                control = float(control.flatten()[0]) if control.size > 0 else None
            elif control is not None:
                control = float(control)
            return control

        return None

    def _occupy_slot_from_neighbor(self, valence: SporeValence, neighbor: Dict[str, Any], distance: int) -> None:
        """
        Заполняет слот валентности на основе информации о соседе.

        Args:
            valence: Валентность споры для заполнения
            neighbor: Информация о соседе
            distance: Расстояние до соседа (1 или 2)
        """
        if distance == 1:
            # Для детей - просто time_direction и control
            time_dir = neighbor.get('time_direction')
            control = neighbor.get('control')

            # Определяем тип управления (max/min)
            if control is not None:
                control_type = 'max' if control > 0 else 'min'
            else:
                control_type = 'max'  # Дефолт

            # Ищем соответствующий слот
            slot = valence.find_slot_by_parameters(
                slot_type='child',
                time_direction=time_dir,
                control_type=control_type
            )

            if slot:
                slot.occupied = True
                slot.neighbor_id = neighbor.get('target_id')
                slot.dt_value = neighbor.get('dt')
                slot.is_fixed = True  # Существующие связи зафиксированы

        elif distance == 2:
            # Для внуков - два шага
            first_time_dir = neighbor.get('first_time_direction')
            first_control = neighbor.get('first_control')
            second_time_dir = neighbor.get('second_time_direction')

            # Определяем тип управления первого шага
            if first_control is not None:
                first_control_type = 'max' if first_control > 0 else 'min'
            else:
                first_control_type = 'max'  # Дефолт

            # Ищем соответствующий слот
            slot = valence.find_slot_by_parameters(
                slot_type='grandchild',
                time_direction=first_time_dir,
                control_type=first_control_type,
                second_time_direction=second_time_dir
            )

            if slot:
                slot.occupied = True
                slot.neighbor_id = neighbor.get('target_id')
                slot.dt_value = neighbor.get('dt')
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
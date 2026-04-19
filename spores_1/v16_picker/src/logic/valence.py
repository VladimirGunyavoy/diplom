"""
Valence System - Система валентности для отслеживания связей спор
==================================================================

Модуль для отслеживания занятых и свободных валентных мест каждой споры.
Каждая спора имеет 12 валентных слотов:
- 4 слота для детей (соседи 1-го порядка)
- 8 слотов для внуков (соседи 2-го порядка, только с чередованием управления)

ВАЖНО: Для внуков учитываются только пути где управление ЧЕРЕДУЕТСЯ.
Пути с одинаковым управлением (max→max, min→min) не являются валентными слотами.

Структура слотов детей (4 слота):
- forward_max: прямое время, максимальное управление
- forward_min: прямое время, минимальное управление
- backward_max: обратное время, максимальное управление
- backward_min: обратное время, минимальное управление

Структура слотов внуков (8 слотов, только с инверсией управления):
- forward_max_forward_min: 1й шаг (forward, max), 2й шаг (forward, min) ✓
- forward_max_backward_min: 1й шаг (forward, max), 2й шаг (backward, min) ✓
- forward_min_forward_max: 1й шаг (forward, min), 2й шаг (forward, max) ✓
- forward_min_backward_max: 1й шаг (forward, min), 2й шаг (backward, max) ✓
- backward_max_forward_min: 1й шаг (backward, max), 2й шаг (forward, min) ✓
- backward_max_backward_min: 1й шаг (backward, max), 2й шаг (backward, min) ✓
- backward_min_forward_max: 1й шаг (backward, min), 2й шаг (forward, max) ✓
- backward_min_backward_max: 1й шаг (backward, min), 2й шаг (backward, max) ✓

Исключены (нет чередования управления):
- forward_max → forward_max (max→max) ❌
- forward_min → forward_min (min→min) ❌
- backward_max → backward_max (max→max) ❌
- backward_min → backward_min (min→min) ❌
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np


@dataclass
class ValenceSlot:
    """
    Одно валентное место споры.

    Представляет конкретное направление роста дерева от споры:
    - Для детей: определяется временем (forward/backward) и управлением (max/min)
    - Для внуков: два шага, второе управление - инверсия первого

    Attributes:
        slot_type: Тип слота ('child' или 'grandchild')
        time_direction: Направление времени первого шага ('forward' или 'backward')
        control_type: Тип управления первого шага ('max' или 'min')
        second_time_direction: Направление времени второго шага (только для внуков)
        dt_value: Значение времени перехода (если слот занят)
        occupied: Занят ли слот
        neighbor_id: ID соседней споры (если слот занят)
        is_fixed: Зафиксирован ли dt для оптимизатора
    """
    slot_type: str  # 'child' или 'grandchild'

    # Параметры первого шага (для детей и внуков)
    time_direction: str  # 'forward' или 'backward'
    control_type: str  # 'max' или 'min'

    # Параметры второго шага (только для внуков)
    second_time_direction: Optional[str] = None  # 'forward' или 'backward'

    # Данные о занятости
    dt_value: Optional[float] = None  # Суммарное время перехода
    dt_sequence: Optional[List[float]] = None  # Последовательность dt по шагам
    occupied: bool = False  # Занят ли слот
    neighbor_id: Optional[int] = None  # ID соседа

    # Для интеграции с оптимизатором
    is_fixed: bool = False  # Зафиксирован ли dt

    def get_second_control_type(self) -> Optional[str]:
        """
        Возвращает тип управления второго шага (инверсия первого).
        Только для внуков.
        """
        if self.slot_type != 'grandchild':
            return None
        return 'min' if self.control_type == 'max' else 'max'

    def get_slot_name(self) -> str:
        """
        Возвращает человекочитаемое имя слота.

        Примеры:
        - Для детей: 'forward_max', 'backward_min'
        - Для внуков: 'forward_max_backward_min', 'backward_min_forward_max'
        """
        if self.slot_type == 'child':
            return f"{self.time_direction}_{self.control_type}"
        else:  # grandchild
            # Второе управление всегда инверсия первого
            second_control = self.get_second_control_type()
            return (f"{self.time_direction}_{self.control_type}_"
                   f"{self.second_time_direction}_{second_control}")

    def __repr__(self) -> str:
        """Красивое строковое представление слота"""
        status = "🔒" if self.occupied else "🔓"
        if self.dt_sequence:
            formatted = ", ".join(f"{dt:+.6f}" for dt in self.dt_sequence)
            dt_str = f"dt=[{formatted}]"
        elif self.dt_value is not None:
            dt_str = f"dt={self.dt_value:+.6f}"
        else:
            dt_str = "dt=None"
        neighbor_str = f"→{self.neighbor_id}" if self.neighbor_id is not None else ""
        fixed_str = "🔧" if self.is_fixed else ""
        return f"{status}{fixed_str} {self.get_slot_name()}: {dt_str} {neighbor_str}"


@dataclass
class SporeValence:
    """
    Полная валентность одной споры.

    Хранит информацию о всех валентных слотах (4 детей + 8 внуков),
    их занятости и параметрах связей.

    ВАЖНО: Для внуков учитываются только слоты с чередованием управления,
    пути с одинаковым управлением (max→max, min→min) не являются валентными.

    Attributes:
        spore_id: ID споры
        total_slots: Общее количество слотов (4 детей + 8 внуков)
        children_slots: Список из 4 слотов для детей
        grandchildren_slots: Список из 8 слотов для внуков (только с чередованием управления)
    """
    spore_id: int
    total_slots: int = 12  # 4 детей + 8 внуков (только с чередованием управления)

    # Слоты детей (4 штуки)
    children_slots: List[ValenceSlot] = field(default_factory=list)

    # Слоты внуков (8 штук)
    grandchildren_slots: List[ValenceSlot] = field(default_factory=list)

    def __post_init__(self):
        """Инициализирует пустые слоты если они не были переданы"""
        if not self.children_slots:
            self.children_slots = self._create_children_slots()
        if not self.grandchildren_slots:
            self.grandchildren_slots = self._create_grandchildren_slots()
        # Обновляем общее количество слотов согласно фактическому числу
        self.total_slots = len(self.children_slots) + len(self.grandchildren_slots)


    def _create_children_slots(self) -> List[ValenceSlot]:
        """Создает 4 пустых слота для детей"""
        return [
            ValenceSlot('child', 'forward', 'max'),
            ValenceSlot('child', 'forward', 'min'),
            ValenceSlot('child', 'backward', 'max'),
            ValenceSlot('child', 'backward', 'min'),
        ]

    def _create_grandchildren_slots(self) -> List[ValenceSlot]:
        """Создает пустые слоты для внуков с чередованием управления"""
        # Только пути с чередованием управления (max→min или min→max)
        slots: List[ValenceSlot] = []

        combinations = [
            ('forward', 'max', 'forward'),
            ('forward', 'max', 'backward'),
            ('forward', 'min', 'forward'),
            ('forward', 'min', 'backward'),
            ('backward', 'max', 'forward'),
            ('backward', 'max', 'backward'),
            ('backward', 'min', 'forward'),
            ('backward', 'min', 'backward'),
        ]

        for time_direction, control_type, second_time_direction in combinations:
            slots.append(ValenceSlot(
                slot_type='grandchild',
                time_direction=time_direction,
                control_type=control_type,
                second_time_direction=second_time_direction
            ))

        return slots

    def get_free_slots(self) -> List[ValenceSlot]:
        """Возвращает список свободных слотов"""
        return [slot for slot in self.children_slots + self.grandchildren_slots
                if not slot.occupied]

    def get_occupied_slots(self) -> List[ValenceSlot]:
        """Возвращает список занятых слотов"""
        return [slot for slot in self.children_slots + self.grandchildren_slots
                if slot.occupied]

    def get_fixed_dt_values(self) -> Dict[str, float]:
        """
        Возвращает словарь зафиксированных dt для передачи в оптимизатор.

        Ключ - имя слота, значение - dt.
        Это dt которые оптимизатор НЕ должен трогать.

        Returns:
            Словарь {имя_слота: dt_value}
        """
        fixed_dt = {}
        for slot in self.get_occupied_slots():
            if slot.is_fixed and slot.dt_value is not None:
                fixed_dt[slot.get_slot_name()] = slot.dt_value
        return fixed_dt

    def get_free_slot_names(self) -> List[str]:
        """
        Возвращает имена свободных слотов для оптимизации.

        Эти слоты оптимизатор ДОЛЖЕН заполнить.

        Returns:
            Список имен свободных слотов
        """
        return [slot.get_slot_name() for slot in self.get_free_slots()]

    def count_free_children(self) -> int:
        """Возвращает количество свободных слотов детей"""
        return sum(1 for slot in self.children_slots if not slot.occupied)

    def count_free_grandchildren(self) -> int:
        """Возвращает количество свободных слотов внуков"""
        return sum(1 for slot in self.grandchildren_slots if not slot.occupied)

    def count_occupied_children(self) -> int:
        """Возвращает количество занятых слотов детей"""
        return sum(1 for slot in self.children_slots if slot.occupied)

    def count_occupied_grandchildren(self) -> int:
        """Возвращает количество занятых слотов внуков"""
        return sum(1 for slot in self.grandchildren_slots if slot.occupied)

    def find_slot_by_name(self, slot_name: str) -> Optional[ValenceSlot]:
        """
        Находит слот по его имени.

        Args:
            slot_name: Имя слота (например, 'forward_max' или 'forward_max_backward_min')

        Returns:
            Найденный слот или None
        """
        for slot in self.children_slots + self.grandchildren_slots:
            if slot.get_slot_name() == slot_name:
                return slot
        return None

    def find_slot_by_parameters(self,
                                slot_type: str,
                                time_direction: str,
                                control_type: str,
                                second_time_direction: Optional[str] = None) -> Optional[ValenceSlot]:
        """
        Находит слот по его параметрам.

        Args:
            slot_type: 'child' или 'grandchild'
            time_direction: 'forward' или 'backward'
            control_type: 'max' или 'min'
            second_time_direction: 'forward' или 'backward' (только для внуков)

        Returns:
            Найденный слот или None
        """
        slots = self.children_slots if slot_type == 'child' else self.grandchildren_slots

        for slot in slots:
            if (slot.time_direction == time_direction and
                slot.control_type == control_type):

                # Для детей этого достаточно
                if slot_type == 'child':
                    return slot

                # Для внуков проверяем второе направление
                if slot.second_time_direction == second_time_direction:
                    return slot

        return None

    def print_summary(self, spore_visual_id: Optional[str] = None, id_mapper = None) -> None:
        """
        Выводит красивый отчет о валентности споры.

        Args:
            spore_visual_id: Визуальный ID споры (как на картинке)
            id_mapper: Функция для конвертации внутренних ID в визуальные (принимает str, возвращает str)
        """
        display_id = spore_visual_id if spore_visual_id else self.spore_id

        print(f"\n📊 ВАЛЕНТНОСТЬ СПОРЫ {display_id}:")
        print(f"   Всего слотов: {self.total_slots}")
        print(f"   Занято: {len(self.get_occupied_slots())}")
        print(f"   Свободно: {len(self.get_free_slots())}")

        print(f"\n   👶 ДЕТИ ({len(self.children_slots)} слотов, "
              f"занято {self.count_occupied_children()}, "
              f"свободно {self.count_free_children()}):")
        for slot in self.children_slots:
            self._print_slot(slot, id_mapper)

        print(f"\n   👶👶 ВНУКИ ({len(self.grandchildren_slots)} слотов, "
              f"занято {self.count_occupied_grandchildren()}, "
              f"свободно {self.count_free_grandchildren()}) "
              f"[только с чередованием управления]:")
        for slot in self.grandchildren_slots:
            self._print_slot(slot, id_mapper)

    def _print_slot(self, slot: ValenceSlot, id_mapper = None) -> None:
        """
        Выводит информацию о слоте с визуальными ID.

        Args:
            slot: Слот для вывода
            id_mapper: Функция для конвертации ID
        """
        status = "🔒" if slot.occupied else "🔓"
        dt_str = f"dt={slot.dt_value:+.6f}" if slot.dt_value is not None else "dt=None"

        # Конвертируем neighbor_id в визуальный если есть mapper
        if slot.neighbor_id is not None:
            if id_mapper and callable(id_mapper):
                visual_neighbor_id = id_mapper(slot.neighbor_id)
                neighbor_str = f"→{visual_neighbor_id}"
            else:
                neighbor_str = f"→{slot.neighbor_id}"
        else:
            neighbor_str = ""

        fixed_str = "🔧" if slot.is_fixed else ""
        print(f"      {status}{fixed_str} {slot.get_slot_name()}: {dt_str} {neighbor_str}")

    def print_optimizer_data(self) -> None:
        """Выводит данные для передачи в оптимизатор"""
        print("\n🔧 ДАННЫЕ ДЛЯ ОПТИМИЗАТОРА:")

        fixed_dt = self.get_fixed_dt_values()
        free_slots = self.get_free_slot_names()

        print(f"   🔒 Зафиксированные dt ({len(fixed_dt)}):")
        if fixed_dt:
            for slot_name, dt in fixed_dt.items():
                print(f"      {slot_name}: {dt:+.6f}")
        else:
            print(f"      (нет зафиксированных)")

        print(f"\n   🔓 Слоты для оптимизации ({len(free_slots)}):")
        if free_slots:
            for slot_name in free_slots:
                print(f"      {slot_name}")
        else:
            print(f"      (нет свободных слотов)")
import numpy as np
from typing import Optional


class TreeOptimizer:
    """
    Класс для оптимизации деревьев и управления dt векторами.
    Отвечает за сохранение и применение оптимизированных времен.
    """

    def __init__(self, pendulum, config):
        """
        Инициализация оптимизатора деревьев.

        Args:
            pendulum: Система маятника
            config: Конфигурация
        """
        self.pendulum = pendulum
        self.config = config

        # Оптимальные dt векторы для деревьев
        self.ghost_tree_dt_vector: Optional[np.ndarray] = None
        self.ghost_tree_optimized: bool = False

        # Отслеживание изменений dt
        self.last_known_dt: Optional[float] = None
        self.tree_created_with_dt: Optional[float] = None

        # Режим отладки
        self.debug_mode: bool = False

    def update_ghost_tree_with_optimal_pairs(self, current_ghost_tree):
        """Обновляет призрачное дерево оптимальными dt из найденных пар."""
        if not current_ghost_tree:
            print("⚠️ Нет призрачного дерева для обновления")
            return

        try:
            from ...logic.tree.pairs.find_optimal_pairs import find_optimal_pairs
            from ...logic.tree.pairs.extract_optimal_times_from_pairs import extract_optimal_times_from_pairs

            print("🔍 Поиск оптимальных пар для текущего призрачного дерева...")

            # Ищем оптимальные пары в уже созданном призрачном дереве
            pairs = find_optimal_pairs(current_ghost_tree, show=True)

            if pairs is None or len(pairs) == 0:
                print("⚠️ Не найдено оптимальных пар")
                return

            print(f"✅ Найдено {len(pairs)} оптимальных пар")

            # Используем правильную функцию для извлечения dt
            dt_results = extract_optimal_times_from_pairs(pairs, current_ghost_tree, show=True)

            if dt_results is None:
                print("❌ Ошибка извлечения оптимальных времен")
                return

            # Получаем оптимизированные dt из результата
            optimal_dt_children = dt_results['dt_children']
            optimal_dt_grandchildren = dt_results['dt_grandchildren']

            print(f"🔍 РЕЗУЛЬТАТ ИЗВЛЕЧЕНИЯ:")
            print(f"   Обновлено внуков: {dt_results['stats']['changed_count']}/{dt_results['stats']['total_grandchildren']}")
            print(f"   Пар использовано: {len(pairs)}")
            print(f"   Неспаренных внуков: {len(dt_results['unpaired_grandchildren'])}")

            # Обновляем сохраненный dt вектор
            old_dt_vector = self.ghost_tree_dt_vector.copy() if self.ghost_tree_dt_vector is not None else None
            self.ghost_tree_dt_vector = np.hstack([optimal_dt_children, optimal_dt_grandchildren])

            print(f"🔍 СРАВНЕНИЕ dt векторов:")
            if old_dt_vector is not None:
                print(f"   Старый: {old_dt_vector}")
            print(f"   Новый:  {self.ghost_tree_dt_vector}")

            # Блокируем автообновление при движении мыши
            self.ghost_tree_optimized = True

            # Фиксируем текущий dt для отслеживания изменений
            self.last_known_dt = self.get_current_dt()

            print(f"🎯 Призрачное дерево обновлено оптимальными dt из {len(pairs)} пар!")
            print(f"   📊 Изменено dt у {dt_results['stats']['changed_count']} внуков")

        except Exception as e:
            print(f"❌ Ошибка обновления призрачного дерева: {e}")
            import traceback
            traceback.print_exc()

    def reset_ghost_tree_optimization(self):
        """Сбрасывает блокировку оптимизированного призрачного дерева."""
        self.ghost_tree_optimized = False
        self.ghost_tree_dt_vector = None
        self.tree_created_with_dt = None
        print("🔓 Оптимизация призрачного дерева сброшена")

    def check_dt_changed(self) -> bool:
        """
        Проверяет изменился ли dt и обновляет состояние.

        Returns:
            True если dt изменился
        """
        current_dt = self.get_current_dt()
        dt_changed = False

        if self.last_known_dt is not None:
            if abs(current_dt - self.last_known_dt) > 1e-10:  # dt изменился
                dt_changed = True
                if self.debug_mode:
                    print(f"🔄 dt изменился: {self.last_known_dt:.6f} → {current_dt:.6f}")

                # Если была оптимизация - сбрасываем её
                if self.ghost_tree_optimized:
                    if self.debug_mode:
                        print(f"🔓 Сброс оптимизации из-за изменения dt")
                    self.ghost_tree_optimized = False
                    self.ghost_tree_dt_vector = None
                    self.tree_created_with_dt = None

        # Сохраняем текущий dt для следующего раза
        self.last_known_dt = current_dt
        return dt_changed

    def should_use_optimized_dt(self, dt_changed: bool) -> bool:
        """
        Определяет использовать ли оптимизированные dt.

        Args:
            dt_changed: Флаг изменения dt

        Returns:
            True если следует использовать оптимизированные dt
        """
        return (self.ghost_tree_optimized and
                self.ghost_tree_dt_vector is not None and
                not dt_changed)

    def get_optimized_dt_vectors(self):
        """
        Возвращает оптимизированные dt векторы.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (dt_children, dt_grandchildren)
        """
        if self.ghost_tree_dt_vector is None or len(self.ghost_tree_dt_vector) != 12:
            return None, None

        dt_children = self.ghost_tree_dt_vector[:4]
        dt_grandchildren = self.ghost_tree_dt_vector[4:12]

        return dt_children, dt_grandchildren

    def save_dt_vector_from_tree(self, tree_logic, dt: float):
        """
        Сохраняет dt вектор из логического дерева.

        Args:
            tree_logic: Логическое дерево
            dt: Базовый dt
        """
        if not hasattr(tree_logic, 'children') or not hasattr(tree_logic, 'grandchildren'):
            return

        if self.ghost_tree_dt_vector is not None:
            return  # Уже есть сохраненный вектор

        try:
            # Извлекаем dt из дерева
            dt_children = [child.get('dt', dt) for child in tree_logic.children]

            # Для внуков используем dt их родителя, умноженный на 0.2
            dt_grandchildren = []
            for gc in tree_logic.grandchildren:
                parent_idx = gc.get('parent_idx', 0)
                parent_dt = dt_children[parent_idx] if parent_idx < len(dt_children) else dt
                dt_grandchildren.append(gc.get('dt', parent_dt * 0.2))

            self.ghost_tree_dt_vector = np.hstack([dt_children, dt_grandchildren])

            # Отладка: показываем что сохраняем
            if self.debug_mode:
                print(f"💾 Сохраняем ghost_tree_dt_vector:")
                print(f"   dt_children: {dt_children}")
                print(f"   dt_grandchildren: {dt_grandchildren}")
                print(f"   ghost_tree_dt_vector: {self.ghost_tree_dt_vector}")

        except Exception as e:
            print(f"⚠️ Ошибка сохранения dt вектора: {e}")

    def get_current_dt(self) -> float:
        """Получает текущий dt из DTManager или конфига."""
        # Попытка получить из DTManager (должен быть передан извне)
        if hasattr(self, 'dt_manager') and self.dt_manager:
            if hasattr(self.dt_manager, 'get_current_dt'):
                return self.dt_manager.get_current_dt()
            elif hasattr(self.dt_manager, 'get_dt'):
                return self.dt_manager.get_dt()

        # Fallback на конфиг
        return self.config.get('pendulum', {}).get('dt', 0.1)

    def set_dt_manager(self, dt_manager):
        """Устанавливает ссылку на DTManager."""
        self.dt_manager = dt_manager

    def set_debug_mode(self, debug: bool):
        """Включает/выключает режим отладки."""
        self.debug_mode = debug

    def get_stats(self) -> dict:
        """Возвращает статистику оптимизатора."""
        return {
            'has_optimized_vector': self.ghost_tree_dt_vector is not None,
            'is_optimized': self.ghost_tree_optimized,
            'last_known_dt': self.last_known_dt,
            'tree_created_with_dt': self.tree_created_with_dt,
            'vector_length': len(self.ghost_tree_dt_vector) if self.ghost_tree_dt_vector is not None else 0
        }

    # Методы для обратной совместимости
    def set_ghost_tree_dt_vector(self, dt_vector):
        """Устанавливает призрачный dt вектор."""
        self.ghost_tree_dt_vector = dt_vector

    def get_ghost_tree_dt_vector(self):
        """Возвращает призрачный dt вектор."""
        return self.ghost_tree_dt_vector

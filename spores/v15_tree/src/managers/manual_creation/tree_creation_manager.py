from typing import Optional, List
import numpy as np
from ursina import destroy

from ...managers.zoom_manager import ZoomManager
from ...managers.spore_manager import SporeManager
from ...managers.color_manager import ColorManager
from ...logic.pendulum import PendulumSystem
from ...core.spore import Spore
from ...visual.link import Link


class TreeCreationManager:
    """
    Управляет созданием деревьев разной глубины.

    Ответственности:
    - Создание деревьев споров в позиции курсора
    - Управление режимами создания (spores/tree)
    - Настройка глубины дерева
    - Интеграция с SporeTree логикой
    """

    def __init__(self,
                 spore_manager: SporeManager,
                 zoom_manager: ZoomManager,
                 pendulum: PendulumSystem,
                 color_manager: ColorManager,
                 config: dict):

        self.spore_manager = spore_manager
        self.zoom_manager = zoom_manager
        self.pendulum = pendulum
        self.color_manager = color_manager
        self.config = config

        # Настройки создания
        self.creation_mode = 'spores'  # 'spores' или 'tree'
        self.tree_depth = 2

        # Сохранение dt вектора от призрачного дерева
        self.ghost_tree_dt_vector = None

        print("   ✓ Tree Creation Manager создан")

    def toggle_creation_mode(self):
        """Переключает режим создания."""
        if self.creation_mode == 'spores':
            self.creation_mode = 'tree'
            print("🌲 Режим: Создание деревьев")
        else:
            self.creation_mode = 'spores'
            print("🌟 Режим: Создание спор")

    def set_tree_depth(self, depth: int):
        """Устанавливает глубину дерева."""
        self.tree_depth = max(1, min(depth, 2))
        print(f"🌲 Глубина дерева: {self.tree_depth}")

    def create_tree_at_cursor(self, preview_position_2d: np.ndarray) -> Optional[List[Spore]]:
        """
        Создает дерево в позиции курсора.

        Args:
            preview_position_2d: Позиция курсора в 2D координатах

        Returns:
            Список созданных спор или None при ошибке

        ЛОГИКА: дерево → споры и линки → добавляем в общий граф → забываем дерево
        """
        print("🚨 ВЫЗВАН create_tree_at_cursor()!!! Начинаем создание дерева!")
        try:
            from ...visual.spore_tree_visual import SporeTreeVisual
            from ...logic.tree.spore_tree import SporeTree
            from ...logic.tree.spore_tree_config import SporeTreeConfig

            # Получаем текущий dt
            dt = self._get_current_dt()

            # Правильная позиция для дерева
            tree_position = np.array([preview_position_2d[0], preview_position_2d[1]])

            # Создаем логику дерева с правильными dt векторами
            if self.ghost_tree_dt_vector is not None and len(self.ghost_tree_dt_vector) == 12:
                # Берем абсолютные значения, т.к. SporeTree ожидает положительные dt
                dt_children_abs = np.abs(self.ghost_tree_dt_vector[:4])
                dt_grandchildren_abs = np.abs(self.ghost_tree_dt_vector[4:])

                print(f"🌲 Используем dt из призрачного дерева")
                print(f"   📊 dt_children: {dt_children_abs}")
                print(f"   📊 dt_grandchildren: {dt_grandchildren_abs}")

                tree_config = SporeTreeConfig(
                    initial_position=tree_position,
                    dt_base=dt
                )

                tree_logic = SporeTree(
                    pendulum=self.pendulum,
                    config=tree_config,
                    dt_children=dt_children_abs,
                    dt_grandchildren=dt_grandchildren_abs,
                    auto_create=False
                )
                
                # Создаем детей (всегда)
                tree_logic.create_children()
                
                # Создаем внуков если нужна глубина 2
                if self.tree_depth >= 2:
                    tree_logic.create_grandchildren()
            else:
                # Создаем обычное дерево с автоматическими dt
                print(f"🌲 Создаем автоматическое дерево")

                tree_config = SporeTreeConfig(
                    initial_position=tree_position,
                    dt_base=dt
                )

                tree_logic = SporeTree(
                    pendulum=self.pendulum,
                    config=tree_config,
                    auto_create=True
                )
                
                # Создаем детей (всегда)
                tree_logic.create_children()
                
                # Создаем внуков если нужна глубина 2
                if self.tree_depth >= 2:
                    tree_logic.create_grandchildren()

            # Создаем визуализацию дерева
            goal_position = self.config.get('spore', {}).get('goal_position', [0, 0])
            
            # DEBUG: Диагностика конфигурации
            print(f"🔍 DEBUG: Полный self.config['spore']: {self.config.get('spore', 'НЕТ КЛЮЧА!')}")
            spore_config = self.config.get('spore', {})
            print(f"🔍 DEBUG: spore_config: {spore_config}")
            spore_scale = spore_config.get('scale', 'НЕТ SCALE!')
            print(f"🔍 DEBUG: spore scale из конфигурации: {spore_scale}")
            print(f"🔍 DEBUG: zoom_manager.spores_scale: {self.zoom_manager.spores_scale}")

            # ПРОВЕРКА: Если scale отсутствует - это баг!
            if 'scale' not in spore_config:
                print(f"❌ БАГ: scale отсутствует в spore_config! Принудительно устанавливаем 0.02")
                spore_config = spore_config.copy()
                spore_config['scale'] = 0.02

            # ВАЖНО: Передаем весь config, не только spore_config, чтобы включить goal_position
            visual_config = self.config.copy()
            visual_config['spore']['goal_position'] = goal_position

            # Правильный конструктор SporeTreeVisual
            tree_visual = SporeTreeVisual(
                color_manager=self.color_manager,
                zoom_manager=self.zoom_manager,
                config=visual_config
            )

            # Устанавливаем логику дерева
            tree_visual.set_tree_logic(tree_logic)
            
            # Создаем визуализацию
            tree_visual.create_visual()

            # Извлекаем споры и линки из дерева
            created_spores = []
            created_links = []

            # Собираем споры
            if tree_visual.root_spore:
                created_spores.append(tree_visual.root_spore)

            created_spores.extend(tree_visual.child_spores)

            if self.tree_depth >= 2:
                created_spores.extend(tree_visual.grandchild_spores)

            # ПРАВИЛЬНЫЙ ПОРЯДОК: Сначала добавляем в систему, потом применяем трансформации
            # 1. Добавляем споры в spore_manager
            for spore in created_spores:
                if spore:
                    self.spore_manager.add_spore_manual(spore)

            # 2. Регистрируем споры в zoom_manager (уникальные имена)
            spore_keys = []
            for spore in created_spores:
                if spore:
                    key = self.zoom_manager.get_unique_spore_id()
                    self.zoom_manager.register_object(spore, key)
                    spore_keys.append(key)
                    spore._zoom_manager_key = key

            # Собираем линки
            created_links.extend(tree_visual.child_links)

            if self.tree_depth >= 2:
                created_links.extend(tree_visual.grandchild_links)

            # 3. Регистрируем линки в zoom_manager (уникальные имена)
            link_keys = []
            for link in created_links:
                if link:
                    key = self.zoom_manager.get_unique_link_id()
                    self.zoom_manager.register_object(link, key)
                    link_keys.append(key)
                    link._zoom_manager_key = key

            # 4. Применяем трансформации ко всем объектам сразу
            self.zoom_manager.update_transform()

            # Освобождаем SporeTreeVisual
            tree_visual.root_spore = None
            tree_visual.child_spores.clear()
            tree_visual.grandchild_spores.clear()
            tree_visual.child_links.clear()
            tree_visual.grandchild_links.clear()
            tree_visual.visual_created = False

            tree_visual = None
            tree_logic = None

            print(f"🌲 Дерево создано в ({preview_position_2d[0]:.3f}, {preview_position_2d[1]:.3f})")
            print(f"   📊 Глубина: {self.tree_depth}, dt: {dt:.4f}")
            print(f"   🎯 Создано: {len(created_spores)} спор + {len(created_links)} линков")

            return created_spores

        except Exception as e:
            print(f"❌ Ошибка создания дерева: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_current_dt(self) -> float:
        """Получает текущий dt из конфигурации."""
        return self.config.get('pendulum', {}).get('dt', 0.1)

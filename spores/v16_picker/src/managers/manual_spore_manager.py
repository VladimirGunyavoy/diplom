from typing import Optional, Tuple, List
import numpy as np
from ursina import Vec3, color, destroy, mouse, camera, raycast, Vec2

from ..core.spore import Spore
from ..managers.zoom_manager import ZoomManager
from ..managers.spore_manager import SporeManager
from ..managers.color_manager import ColorManager
from ..managers.manual_creation.preview_manager import PreviewManager
from ..managers.manual_creation.prediction_manager import PredictionManager
from ..managers.manual_creation.tree_creation_manager import TreeCreationManager
from ..logic.pendulum import PendulumSystem
from ..visual.link import Link


class ManualSporeManager:
    """
    Менеджер для ручного создания спор с превью по курсору.

    Функциональность:
    - Показывает полупрозрачную спору в позиции курсора мыши
    - Отображает предсказания при min/max управлении
    - Создает споры + цепочки родителей по ЛКМ
    """

    def __init__(self,
                 spore_manager: SporeManager,
                 zoom_manager: ZoomManager,
                 pendulum: PendulumSystem,
                 color_manager: ColorManager,
                 config: dict):

        self.spore_manager = spore_manager

        # Создаем SharedDependencies один раз
        from .manual_creation.shared_dependencies import SharedDependencies
        self.deps = SharedDependencies(
            zoom_manager=zoom_manager,
            color_manager=color_manager,
            pendulum=pendulum,
            config=config
        )

        # Создаем подкомпоненты с общими зависимостями
        self.preview_manager = PreviewManager(self.deps)
        self.prediction_manager = PredictionManager(self.deps) 
        self.tree_creation_manager = TreeCreationManager(self.deps, spore_manager)
        
        # Синхронизируем ghost_tree_dt_vector
        # self.tree_creation_manager.ghost_tree_dt_vector = self._ghost_tree_dt_vector

        # Общие созданные линки
        self.created_links: List[Link] = []
        
        # Получаем id_manager от spore_manager
        self.id_manager = self.spore_manager.id_manager

        # Сохранение dt вектора от призрачного дерева
        self._ghost_tree_dt_vector = None

        # Базовый dt на момент вычисления пар (для последующего масштабирования при прокрутке)
        self.ghost_dt_baseline: Optional[float] = None

        # История созданных групп спор для возможности удаления
        self.spore_groups_history: List[List[Spore]] = []  # История групп спор
        self.group_links_history: List[List[Link]] = []    # История линков для каждой группы

        # Собственные поля состояния (без синхронизации)
        self.creation_mode = 'spores'  # Только в ManualSporeManager
        self.tree_depth = 2           # Только в ManualSporeManager

        print(f"   ✓ Manual Spore Manager создан (управление: {self.prediction_manager.min_control} .. {self.prediction_manager.max_control})")
        print(f"   📚 История групп инициализирована")

    def toggle_creation_mode(self):
        """Переключает режим создания."""
        self.creation_mode = 'tree' if self.creation_mode == 'spores' else 'spores'
        mode_name = 'деревья' if self.creation_mode == 'tree' else 'споры'
        print(f"🔄 Режим создания: {mode_name}")

    def set_tree_depth(self, depth: int):
        """Устанавливает глубину дерева."""
        self.tree_depth = max(1, min(depth, 2))
        print(f"🌲 Глубина дерева: {self.tree_depth}")



    @property
    def ghost_tree_dt_vector(self):
        """Получает dt вектор от призрачного дерева."""
        return self._ghost_tree_dt_vector

    @ghost_tree_dt_vector.setter
    def ghost_tree_dt_vector(self, value):
        """Устанавливает dt вектор и синхронизирует с TreeCreationManager."""
        self._ghost_tree_dt_vector = value
        if hasattr(self, 'tree_creation_manager'):
            self.tree_creation_manager.ghost_tree_dt_vector = value
            




    def _get_next_link_id(self) -> int:
        """Возвращает уникальный ID для линка"""
        return self.id_manager.get_next_link_id()

    def _get_next_spore_id(self) -> int:
        """Возвращает уникальный ID для споры"""
        return self.id_manager.get_next_spore_id()

    def get_mouse_world_position(self) -> Optional[Tuple[float, float]]:
        """Получает позицию курсора мыши."""
        pos_2d = self.deps.get_cursor_position_2d()
        return (pos_2d[0], pos_2d[1])

    def update_cursor_position(self) -> None:
        """
        Обновляет позицию превью споры по позиции курсора мыши.
        """
            # print(f"DEBUG: update_cursor_position вызван")
            # print(f"   preview_enabled: {self.preview_manager.preview_enabled}")
        
        if not self.preview_manager.preview_enabled:
            # print("   STOP: preview отключен")  # Убрано спам-сообщение
            return

        # Получаем правильную позицию курсора мыши
        mouse_pos = self.get_mouse_world_position()
        # print(f"   mouse_pos: {mouse_pos}")
        
        if mouse_pos is None:
            # print("   STOP: mouse_pos = None")  # Убрано спам-сообщение
            return

        # Используем PreviewManager для обновления позиции
        # print("   Вызываем preview_manager.update_cursor_position...")
        self.preview_manager.update_cursor_position(mouse_pos)

        # Обновляем предсказания
        # print(f"   show_predictions: {self.prediction_manager.show_predictions}")
        if self.prediction_manager.show_predictions:
            # print("   Вызываем _update_predictions...")
            self._update_predictions()
        else:
            # print("   SKIP: предсказания отключены")  # Убрано спам-сообщение
            pass

    def _update_predictions(self) -> None:
        """Обновляет предсказания в зависимости от режима создания."""
        self.prediction_manager.update_predictions(
            preview_spore=self.preview_manager.get_preview_spore(),
            preview_position_2d=self.preview_manager.get_preview_position(),
            creation_mode=self.creation_mode,
            tree_depth=self.tree_depth,
            ghost_dt_vector=self._ghost_tree_dt_vector
        )

    def create_spore_at_cursor(self):
        """Создает дерево нужной глубины в зависимости от режима."""
        if not self.preview_manager.preview_enabled or not self.preview_manager.has_preview():
            return None

        preview_position_2d = self.preview_manager.get_preview_position()
        
        # Определяем глубину дерева по режиму
        if self.creation_mode == 'tree':
            depth = self.tree_depth  # 1 или 2
        else:
            depth = 1  # "spores" режим = дерево глубины 1
        
        # Используем TreeCreationManager для всех случаев
        created_spores = self.tree_creation_manager.create_tree_at_cursor(preview_position_2d, depth)
        
        if created_spores:
            # Сохраняем в историю (логика остается прежней)
            created_links = []  # TreeCreationManager уже обработал линки
            self.spore_groups_history.append(created_spores.copy())
            self.group_links_history.append(created_links.copy())
            print(f"   📚 Группа #{len(self.spore_groups_history)} сохранена в истории")
            
        return created_spores



    def toggle_ghost_system(self):
        """
        Переключает видимость всей призрачной системы.
        Управляет preview_manager и prediction_manager синхронно.
        
        Returns:
            bool: новое состояние (True = включено)
        """
        # Получаем текущее состояние от preview_manager
        current_state = self.preview_manager.preview_enabled
        new_state = not current_state
        
        # Переключаем preview spore
        self.preview_manager.preview_enabled = new_state
        
        # Переключаем predictions
        self.prediction_manager.show_predictions = new_state
        
        # Если отключаем - очищаем все призрачные объекты
        if not new_state:
            self.prediction_manager.clear_predictions()
            # Скрываем preview spore если существует
            if self.preview_manager.preview_spore:
                self.preview_manager.preview_spore.visible = False
        else:
            # Если включаем - показываем preview spore обратно
            if self.preview_manager.preview_spore:
                self.preview_manager.preview_spore.visible = True
        
        # Информативный вывод
        state_name = "включена" if new_state else "отключена"
        print(f"👻 Призрачная система {state_name}")
        
        return new_state

    def _destroy_preview(self) -> None:
        """Уничтожает предсказания и их линки."""
        self.prediction_manager.clear_predictions()

    def destroy(self) -> None:
        """Очищает все ресурсы менеджера."""
        self.preview_manager.destroy()
        self._destroy_preview()
        print("   ✓ Manual Spore Manager уничтожен")

    def clear_all(self) -> None:
        """Очищает все ресурсы созданные ManualSporeManager."""
        print("🧹 ManualSporeManager: очистка всех ресурсов...")

        # ДОБАВИТЬ ПЕРЕД существующей логикой:
        # Очищаем объекты TreeCreationManager
        self.tree_creation_manager.clear_all_created_objects()

        # 🆕 ДИАГНОСТИКА: Проверяем состояние до удаления
        print(f"   📊 Линков для удаления: {len(self.created_links)}")

        # 1. Уничтожаем все созданные линки
        for i, link in enumerate(self.created_links):
            try:
                # 🆕 ДИАГНОСТИКА: Проверяем состояние линка
                print(f"   🔍 Линк {i+1}: enabled={getattr(link, 'enabled', 'N/A')}, visible={getattr(link, 'visible', 'N/A')}")
                print(f"            parent={getattr(link, 'parent', 'N/A')}, model={getattr(link, 'model', 'N/A')}")

                # Пробуем разные способы удаления
                link.enabled = False  # Отключаем
                link.visible = False  # Скрываем
                link.parent = None    # type: ignore # Отвязываем от родителя

                destroy(link)  # Уничтожаем
                print(f"   ✅ Линк {i+1} обработан")

            except Exception as e:
                print(f"   ❌ Ошибка с линком {i+1}: {e}")

        # 🆕 ДИАГНОСТИКА: Проверяем глобальное состояние Ursina
        try:
            from ursina import scene, camera
            all_entities = [e for e in scene.entities if hasattr(e, 'model') and e.model]
            arrow_entities = [e for e in all_entities if 'arrow' in str(e.model).lower()]
            print(f"   📊 Всего entities в сцене: {len(scene.entities)}")
            print(f"   🏹 Entities со стрелками: {len(arrow_entities)}")
        except Exception as e:
            print(f"   ⚠️ Ошибка диагностики сцены: {e}")

        # 🆕 Очищаем историю групп
        cleared_groups = len(self.spore_groups_history)
        self.spore_groups_history.clear()
        self.group_links_history.clear()
        print(f"   📚 Очищена история: {cleared_groups} групп")

        self.created_links.clear()

    def delete_last_spore_group(self) -> bool:
        """
        Удаляет последнюю созданную группу спор и их линки.

        Returns:
            True если удаление успешно, False если нечего удалять
        """
        if not self.spore_groups_history:
            print("   ⚠️ Нет групп для удаления")
            return False

        try:
            # 1. Получаем последнюю группу из истории
            last_spores = self.spore_groups_history.pop()
            last_links = self.group_links_history.pop()

            print(f"   🗑️ Удаление группы #{len(self.spore_groups_history) + 1}")
            print(f"   📊 К удалению: {len(last_spores)} спор + {len(last_links)} линков")

            # 2. УДАЛЯЕМ ЛИНКИ (важно делать ДО удаления спор)
            deleted_links = 0
            for i, link in enumerate(last_links):
                try:
                    # Дерегистрируем из zoom_manager используя сохраненный ключ
                    if hasattr(link, '_zoom_manager_key'):
                        key = link._zoom_manager_key
                        self.deps.zoom_manager.unregister_object(key)
                        print(f"   ✓ Линк {i+1} дерегистрирован: {key}")
                    else:
                        # Fallback: поиск по ссылке
                        if hasattr(self.deps.zoom_manager, 'objects'):
                            for key, obj in list(self.deps.zoom_manager.objects.items()):
                                if obj is link:
                                    self.deps.zoom_manager.unregister_object(key)
                                    print(f"   ✓ Линк {i+1} дерегистрирован (fallback): {key}")
                                    break

                    # Удаляем из created_links
                    if link in self.created_links:
                        self.created_links.remove(link)

                    # Уничтожаем объект Ursina
                    destroy(link)
                    deleted_links += 1
                    print(f"   ✅ Линк {i+1} удален")

                except Exception as e:
                    print(f"   ❌ Ошибка удаления линка {i+1}: {e}")

            # 3. УДАЛЯЕМ СПОРЫ
            deleted_spores = 0
            for i, spore in enumerate(last_spores):
                try:
                    # Удаляем из SporeManager
                    if hasattr(self.spore_manager, 'remove_spore'):
                        removed = self.spore_manager.remove_spore(spore)
                        if removed:
                            print(f"   ✓ Спора {i+1} удалена из SporeManager")
                    else:
                        # Fallback если remove_spore еще не реализован
                        if hasattr(self.spore_manager, 'objects') and spore in self.spore_manager.objects:
                            self.spore_manager.objects.remove(spore)
                            print(f"   ✓ Спора {i+1} удалена из objects (fallback)")

                    # Дерегистрируем из zoom_manager используя сохраненный ключ
                    if hasattr(spore, '_zoom_manager_key'):
                        key = spore._zoom_manager_key
                        self.deps.zoom_manager.unregister_object(key)
                        print(f"   ✓ Спора {i+1} дерегистрирована: {key}")
                    else:
                        # Fallback: поиск по ссылке
                        if hasattr(self.deps.zoom_manager, 'objects'):
                            for key, obj in list(self.deps.zoom_manager.objects.items()):
                                if obj is spore:
                                    self.deps.zoom_manager.unregister_object(key)
                                    print(f"   ✓ Спора {i+1} дерегистрирована (fallback): {key}")
                                    break

                    # Уничтожаем объект Ursina
                    destroy(spore)
                    deleted_spores += 1
                    print(f"   ✅ Спора {i+1} уничтожена")

                except Exception as e:
                    print(f"   ❌ Ошибка удаления споры {i+1}: {e}")

            # 4. ИТОГОВАЯ СТАТИСТИКА
            print(f"   🎯 УДАЛЕНИЕ ЗАВЕРШЕНО:")
            print(f"      📊 Спор удалено: {deleted_spores}/{len(last_spores)}")
            print(f"      🔗 Линков удалено: {deleted_links}/{len(last_links)}")
            print(f"      📚 Групп осталось в истории: {len(self.spore_groups_history)}")

            # Проверяем что удаление было успешным
            if deleted_spores == len(last_spores) and deleted_links == len(last_links):
                print(f"   ✅ Группа успешно удалена!")
                return True
            else:
                print(f"   ⚠️ Удаление частично неуспешно")
                return False

        except Exception as e:
            print(f"   ❌ Критическая ошибка удаления группы: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_groups_history_stats(self) -> dict:
        """
        Возвращает статистику по истории созданных групп.

        Returns:
            Словарь с информацией о количестве групп, спор и линков в истории
        """
        total_groups = len(self.spore_groups_history)
        total_spores = sum(len(group) for group in self.spore_groups_history)
        total_links = sum(len(links) for links in self.group_links_history)

        return {
            'total_groups': total_groups,
            'total_spores': total_spores,
            'total_links': total_links,
            'can_delete': total_groups > 0
        }

    def print_groups_history_stats(self) -> None:
        """Выводит детальную статистику истории групп."""
        stats = self.get_groups_history_stats()

        print(f"\n📚 СТАТИСТИКА ИСТОРИИ ГРУПП:")
        print(f"   🔢 Всего групп: {stats['total_groups']}")
        print(f"   🧬 Всего спор: {stats['total_spores']}")
        print(f"   🔗 Всего линков: {stats['total_links']}")
        print(f"   🗑️ Можно удалить: {'Да' if stats['can_delete'] else 'Нет'}")

        if stats['total_groups'] > 0:
            print(f"   📋 Последняя группа: {len(self.spore_groups_history[-1])} спор + {len(self.group_links_history[-1])} линков")
        print("========================")

    def _get_current_dt(self) -> float:
        """Получает текущий dt из конфигурации."""
        return self.deps.config.get('pendulum', {}).get('dt', 0.1)
from typing import List, Dict, Any
from ursina import destroy

from ...core.spore import Spore
from ...visual.link import Link
from ...managers.zoom_manager import ZoomManager
from ...managers.spore_manager import SporeManager


class CreationHistory:
    """
    Класс для управления историей созданных групп спор и линков.
    Позволяет отменить последние операции создания.
    """

    def __init__(self,
                 spore_manager: SporeManager,
                 zoom_manager: ZoomManager):
        """
        Инициализация истории создания.

        Args:
            spore_manager: Менеджер спор для удаления объектов
            zoom_manager: Менеджер зума для дерегистрации объектов
        """
        self.spore_manager = spore_manager
        self.zoom_manager = zoom_manager

        # История созданных групп спор и линков
        self.spore_groups_history: List[List[Spore]] = []
        self.group_links_history: List[List[Link]] = []

    def add_group(self, spores: List[Spore], links: List[Link]) -> None:
        """
        Добавляет новую группу спор и линков в историю.

        Args:
            spores: Список созданных спор
            links: Список созданных линков
        """
        self.spore_groups_history.append(spores.copy())
        self.group_links_history.append(links.copy())

        print(f"   📚 Группа #{len(self.spore_groups_history)} добавлена в историю")
        print(f"      📊 {len(spores)} спор + {len(links)} линков")

    def delete_last_group(self) -> bool:
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
            deleted_links = self._delete_links(last_links)

            # 3. УДАЛЯЕМ СПОРЫ
            deleted_spores = self._delete_spores(last_spores)

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

    def _delete_links(self, links: List[Link]) -> int:
        """
        Удаляет список линков.

        Args:
            links: Список линков для удаления

        Returns:
            Количество успешно удаленных линков
        """
        deleted_links = 0
        for i, link in enumerate(links):
            try:
                # Дерегистрируем из zoom_manager
                if hasattr(self.zoom_manager, 'objects'):
                    for key, obj in list(self.zoom_manager.objects.items()):
                        if obj is link:
                            self.zoom_manager.unregister_object(key)
                            print(f"   ✓ Линк {i+1} дерегистрирован: {key}")
                            break

                # Уничтожаем объект Ursina
                destroy(link)
                deleted_links += 1
                print(f"   ✅ Линк {i+1} удален")

            except Exception as e:
                print(f"   ❌ Ошибка удаления линка {i+1}: {e}")

        return deleted_links

    def _delete_spores(self, spores: List[Spore]) -> int:
        """
        Удаляет список спор.

        Args:
            spores: Список спор для удаления

        Returns:
            Количество успешно удаленных спор
        """
        deleted_spores = 0
        for i, spore in enumerate(spores):
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

                # Дерегистрируем из zoom_manager
                if hasattr(self.zoom_manager, 'objects'):
                    for key, obj in list(self.zoom_manager.objects.items()):
                        if obj is spore:
                            self.zoom_manager.unregister_object(key)
                            print(f"   ✓ Спора {i+1} дерегистрирована: {key}")
                            break

                # Уничтожаем объект Ursina
                destroy(spore)
                deleted_spores += 1
                print(f"   ✅ Спора {i+1} уничтожена")

            except Exception as e:
                print(f"   ❌ Ошибка удаления споры {i+1}: {e}")

        return deleted_spores

    def get_stats(self) -> Dict[str, Any]:
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
            'can_delete': total_groups > 0,
            'last_group_spores': len(self.spore_groups_history[-1]) if total_groups > 0 else 0,
            'last_group_links': len(self.group_links_history[-1]) if total_groups > 0 else 0
        }

    def print_stats(self) -> None:
        """Выводит детальную статистику истории групп."""
        stats = self.get_stats()

        print(f"\n📚 СТАТИСТИКА ИСТОРИИ ГРУПП:")
        print(f"   🔢 Всего групп: {stats['total_groups']}")
        print(f"   🧬 Всего спор: {stats['total_spores']}")
        print(f"   🔗 Всего линков: {stats['total_links']}")
        print(f"   🗑️ Можно удалить: {'Да' if stats['can_delete'] else 'Нет'}")

        if stats['total_groups'] > 0:
            print(f"   📋 Последняя группа: {stats['last_group_spores']} спор + {stats['last_group_links']} линков")
        print("========================")

    def clear_all(self) -> None:
        """Очищает всю историю."""
        cleared_groups = len(self.spore_groups_history)
        self.spore_groups_history.clear()
        self.group_links_history.clear()
        print(f"   📚 Очищена история: {cleared_groups} групп")

    def has_groups(self) -> bool:
        """Проверяет есть ли группы в истории."""
        return len(self.spore_groups_history) > 0

    def get_group_count(self) -> int:
        """Возвращает количество групп в истории."""
        return len(self.spore_groups_history)

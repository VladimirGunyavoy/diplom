from typing import Optional, Tuple, List
import numpy as np
from ursina import Vec3, color, destroy, mouse, camera, raycast, Vec2, scene

from ..core.spore import Spore
from ..managers.zoom_manager import ZoomManager
from ..managers.spore_manager import SporeManager
from ..managers.color_manager import ColorManager
from ..logic.pendulum import PendulumSystem
from ..logic.tree.pairs.find_optimal_pairs import find_optimal_pairs
from ..logic.tree.pairs.extract_optimal_times_from_pairs import extract_optimal_times_from_pairs
from ..visual.prediction_visualizer import PredictionVisualizer
from ..visual.link import Link
from .manual_components import CursorTracker, PreviewManager, CreationHistory, SporeCreator, TreeOptimizer


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
        self.zoom_manager = zoom_manager
        self.pendulum = pendulum
        self.color_manager = color_manager
        self.config = config

        # Создаем трекер курсора
        self.cursor_tracker = CursorTracker(zoom_manager)

        # Создаем менеджер превью
        self.preview_manager = PreviewManager(
            cursor_tracker=self.cursor_tracker,
            pendulum=pendulum,
            zoom_manager=zoom_manager,
            color_manager=color_manager,
            config=config
        )

        # Создаем историю создания
        self.creation_history = CreationHistory(
            spore_manager=spore_manager,
            zoom_manager=zoom_manager
        )

                # Создаем создатель спор
        self.spore_creator = SporeCreator(
            cursor_tracker=self.cursor_tracker,
            spore_manager=spore_manager,
            zoom_manager=zoom_manager,
            color_manager=color_manager,
            pendulum=pendulum,
            config=config
        )

        # Создаем оптимизатор деревьев
        self.tree_optimizer = TreeOptimizer(pendulum, config)
        # Передаем ссылку на dt_manager если есть
        if hasattr(self, 'dt_manager'):
            self.tree_optimizer.set_dt_manager(self.dt_manager)
        
        # Настройки превью
        self.debug_mode = False  # Флаг для отладочного вывода
        

        
        # Получаем границы управления из маятника
        control_bounds = self.pendulum.get_control_bounds()
        self.min_control = float(control_bounds[0])
        self.max_control = float(control_bounds[1])









        print(f"   ✓ Manual Spore Manager создан (управление: {self.min_control} .. {self.max_control})")
        print(f"   📚 История групп инициализирована")





        # 🆕 Добавляем поддержку деревьев
        # НЕ НУЖНО: self.creation_mode и self.tree_depth теперь properties!
        # НЕ НУЖНО: self.active_trees = [] - споры идут в общий граф!
    
        print("   🌲 Поддержка деревьев добавлена")

        # Безопасные методы для регистрации/удаления объектов
        self._safe_register = lambda obj, reg_id: self._safe_register_impl(obj, reg_id)
        self._safe_unregister_and_destroy = lambda obj: self._safe_unregister_and_destroy_impl(obj)

    def _safe_register_impl(self, obj, reg_id: str):
        """Безопасная регистрация объекта в zoom_manager."""
        try:
            self.zoom_manager.register_object(obj, reg_id)
            setattr(obj, '_reg_id', reg_id)
        except Exception as e:
            print(f"⚠️ Регистрация {reg_id} не удалась: {e}")

    def _safe_unregister_and_destroy_impl(self, obj):
        """Безопасное удаление объекта с отменой регистрации."""
        try:
            if hasattr(obj, '_reg_id'):
                try:
                    self.zoom_manager.unregister_object(obj._reg_id)
                except:
                    pass
            destroy(obj)
        except Exception as e:
            print(f"⚠️ Удаление {getattr(obj, '_reg_id', '<no_id>')} не удалось: {e}")

# В том же файле добавить методы:










    

    
    def update_cursor_position(self) -> None:
        """
        Обновляет позицию превью споры по позиции курсора мыши.
        """
        # Обновляем позицию курсора через трекер
        if not self.cursor_tracker.update_position():
            return

        # Обновляем превью через preview_manager
        self.preview_manager.update_preview()
    

    

    

    







    def _update_ghost_predictions(self):
        """Обновляет призрачные предсказания дерева."""

        if not self.preview_manager.get_preview_spore():
            return

        # Получаем текущий dt
        current_dt = self._get_current_dt()

        # Проверяем изменения dt через TreeOptimizer
        dt_changed = self.tree_optimizer.check_dt_changed()

        # Определяем режим создания дерева
        use_optimized_dt = self.tree_optimizer.should_use_optimized_dt(dt_changed)

        if use_optimized_dt:
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print("🎯 Используем заблокированные оптимизированные dt")
        else:
            # Выводим сообщение только если dt изменился или это первый вызов (и включен режим отладки)
            if hasattr(self, 'debug_mode') and self.debug_mode:
                if dt_changed:
                    print(f"🔄 Пересоздаем дерево с новым dt: {current_dt:.6f}")
                elif self.tree_optimizer.tree_created_with_dt != current_dt:
                    print(f"🌲 Создаем дерево с dt: {current_dt:.6f}")
                    self.tree_optimizer.tree_created_with_dt = current_dt

        try:
            # Импорты для дерева
            from ..logic.tree.spore_tree import SporeTree
            from ..logic.tree.spore_tree_config import SporeTreeConfig

            # Используем уже полученный current_dt вместо повторного вызова
            dt = current_dt

            # Создаем конфиг дерева
            cursor_pos = self.cursor_tracker.get_current_position()
            tree_config = SporeTreeConfig(
                initial_position=cursor_pos.copy(),
                dt_base=dt,
                dt_grandchildren_factor=0.2,
                show_debug=False
            )

            # ПРОВЕРЯЕМ: используем ли оптимизированные dt
            if use_optimized_dt:
                # Используем сохраненные оптимизированные dt с сохранением знаков
                dt_children = self.tree_optimizer.get_ghost_tree_dt_vector()[:4]  # ✅ ИСПРАВЛЕНИЕ: сохраняем знаки
                dt_grandchildren = self.tree_optimizer.get_ghost_tree_dt_vector()[4:12]  # ✅ ИСПРАВЛЕНИЕ: сохраняем знаки

                # Проверяем размеры векторов перед использованием
                assert len(dt_children) == 4, f"Ожидается 4 dt для детей, получено {len(dt_children)}"
                assert len(dt_grandchildren) == 8, f"Ожидается 8 dt для внуков, получено {len(dt_grandchildren)}"

                # Отладочные принты для проверки что dt действительно применяются
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"🎯 Создаем дерево с оптимизированными dt:")
                    print(f"   dt_children: {dt_children}")
                    print(f"   dt_grandchildren: {dt_grandchildren}")
                    print(f"   Используем ghost_tree_dt_vector: {self.tree_optimizer.get_ghost_tree_dt_vector()}")

                tree_logic = SporeTree(
                    pendulum=self.pendulum,
                    config=tree_config,
                    dt_children=dt_children,  # ✅ ИСПРАВЛЕНИЕ: передаем подписанные dt в конструктор
                    dt_grandchildren=dt_grandchildren,  # ✅ ИСПРАВЛЕНИЕ: передаем подписанные dt в конструктор
                    auto_create=True,  # ✅ ИСПРАВЛЕНИЕ: используем auto_create=True
                    show=False
                )
                # ✅ НЕ вызываем create_children() и create_grandchildren() - дерево уже создано с правильными dt!

                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print("🎯 Используем оптимизированные dt для призрачного дерева")
            else:
                # Создаем обычное дерево с стандартными dt (используем уже полученный current_dt)

                # Стандартные dt: детям - полный dt, внукам - в 5 раз меньше
                dt_children_std = np.array([current_dt, -current_dt, current_dt, -current_dt])  # ✅ с правильными знаками
                dt_grandchildren_std = np.array([
                    current_dt * 0.2, -current_dt * 0.2,  # внуки от child_0
                    current_dt * 0.2, -current_dt * 0.2,  # внуки от child_1
                    current_dt * 0.2, -current_dt * 0.2,  # внуки от child_2
                    current_dt * 0.2, -current_dt * 0.2   # внуки от child_3
                ])

                tree_logic = SporeTree(
                    pendulum=self.pendulum,
                    config=tree_config,
                    dt_children=dt_children_std,  # ✅ ИСПРАВЛЕНИЕ: передаем стандартные dt с правильными знаками
                    dt_grandchildren=dt_grandchildren_std,  # ✅ ИСПРАВЛЕНИЕ: передаем стандартные dt с правильными знаками
                    auto_create=True,  # ✅ ИСПРАВЛЕНИЕ: используем auto_create=True
                    show=False
                )

            # Сохраняем ссылку на призрачное дерево для дальнейшего использования
            self.current_ghost_tree = tree_logic

            # ✅ НЕ вызываем create_children() и create_grandchildren() - дерево уже создано с auto_create=True!

            # Сохраняем dt вектор ТОЛЬКО если его еще нет
            if hasattr(tree_logic, 'children') and hasattr(tree_logic, 'grandchildren'):
                if not hasattr(self, 'ghost_tree_dt_vector') or self.tree_optimizer.get_ghost_tree_dt_vector() is None:
                    try:
                        # Извлекаем dt из дерева
                        dt_children = [child.get('dt', dt) for child in tree_logic.children]
                        # Для внуков используем dt их родителя, умноженный на 0.2
                        dt_grandchildren = []
                        for gc in tree_logic.grandchildren:
                            parent_idx = gc.get('parent_idx', 0)
                            parent_dt = dt_children[parent_idx] if parent_idx < len(dt_children) else dt
                            dt_grandchildren.append(gc.get('dt', parent_dt * 0.2))
                        self.tree_optimizer.set_ghost_tree_dt_vector(np.hstack([dt_children, dt_grandchildren]))

                        # Отладка: показываем что сохраняем
                        if hasattr(self, 'debug_mode') and self.debug_mode:
                            print(f"💾 Сохраняем ghost_tree_dt_vector:")
                            print(f"   dt_children: {dt_children}")
                            print(f"   dt_grandchildren: {dt_grandchildren}")
                            print(f"   ghost_tree_dt_vector: {self.tree_optimizer.get_ghost_tree_dt_vector()}")
                    except Exception as e:
                        print(f"⚠️ Ошибка сохранения dt вектора: {e}")
                        # Не сбрасываем ghost_tree_dt_vector, если он был установлен ранее
                        if not hasattr(self, 'ghost_tree_dt_vector'):
                            self.tree_optimizer.set_ghost_tree_dt_vector(None)

            # Конвертируем в призрачные предсказания через PreviewManager
            self.preview_manager.set_creation_mode('tree')
            self.preview_manager._update_ghost_predictions()

            # Диагностика созданного дерева (только в режиме отладки)
            if hasattr(self, 'debug_mode') and self.debug_mode:
                use_optimized = hasattr(self, 'ghost_tree_dt_vector') and self.tree_optimizer.get_ghost_tree_dt_vector() is not None and not dt_changed
                self._debug_dump_tree("GHOST" if use_optimized else "STD", tree_logic)

        except Exception as e:
            print(f"Ошибка создания призрачного дерева: {e}")

    def _create_ghost_tree_from_logic(self, tree_logic):
        """Создает призрачные споры и линки из логики дерева."""
        # Создаем призрачные споры для детей
        child_ghosts = []
        for i, child_data in enumerate(tree_logic.children):
            ghost_viz = self._create_ghost_spore_from_data(child_data, f"tree_child_{i}", 0.4)
            if ghost_viz and ghost_viz.ghost_spore:
                child_ghosts.append(ghost_viz.ghost_spore)

        # Создаем призрачные споры для внуков (если есть)
        grandchild_ghosts = []
        if hasattr(tree_logic, 'grandchildren') and tree_logic.grandchildren:
            for i, grandchild_data in enumerate(tree_logic.grandchildren):
                ghost_viz = self._create_ghost_spore_from_data(grandchild_data, f"tree_grandchild_{i}", 0.3)
                if ghost_viz and ghost_viz.ghost_spore:
                    grandchild_ghosts.append(ghost_viz.ghost_spore)

        # Создаем призрачные линки от корня к детям с цветом по управлению
        for i, child_ghost in enumerate(child_ghosts):
            if child_ghost and i < len(tree_logic.children):
                # Получаем управление ребенка из данных дерева
                child_control = tree_logic.children[i]['control']

                # Выбираем цвет в зависимости от знака управления
                if child_control >= 0:
                    link_color = 'ghost_max'  # Положительное управление - красный
                else:
                    link_color = 'ghost_min'  # Отрицательное управление - синий

                self._create_ghost_link(
                    self.preview_manager.get_preview_spore(),
                    child_ghost,
                    f"root_to_child_{i}",
                    link_color
                )

        # Создаем призрачные линки от детей к внукам (если tree_depth >= 2)
        if self.tree_depth >= 2 and grandchild_ghosts:
            for i, grandchild_ghost in enumerate(grandchild_ghosts):
                if grandchild_ghost:
                    # Определяем родителя внука из данных
                    grandchild_data = tree_logic.grandchildren[i]
                    parent_idx = grandchild_data['parent_idx']

                    if parent_idx < len(child_ghosts) and child_ghosts[parent_idx]:
                        # Определяем цвет по реальному управлению внука
                        grandchild_control = grandchild_data.get('control', 0.0)
                        link_color = ('ghost_max' if grandchild_control >= 0 else 'ghost_min')

                        self._create_ghost_link(
                            child_ghosts[parent_idx],
                            grandchild_ghost,
                            f"child_{parent_idx}_to_grandchild_{i}",
                            link_color  # Цвет по реальному управлению
                        )

    def _create_ghost_spore_from_data(self, spore_data, name_suffix, alpha):
        """Создает одну призрачную спору из данных дерева."""
        from ..visual.prediction_visualizer import PredictionVisualizer

        # Получаем финальную позицию споры
        final_position = spore_data['position']  # должно быть [угол, угловая_скорость]

        # Конвертируем в numpy array если нужно
        if not isinstance(final_position, np.ndarray):
            final_position = np.array(final_position, dtype=np.float64)

        # Убеждаемся что форма правильная (2,)
        if final_position.shape != (2,):
            final_position = final_position.reshape(2)



        # Создаем визуализатор предсказания
        prediction_viz = PredictionVisualizer(
            parent_spore=self.preview_manager.get_preview_spore(),
            color_manager=self.color_manager,
            zoom_manager=self.zoom_manager,
            cost_function=None,
            config={
                'spore': {'show_ghosts': True},
                'angel': {'show_angels': False, 'show_pillars': False}
            },
            spore_id=f'tree_ghost_{name_suffix}'
        )

        # Устанавливаем позицию призрака
        if prediction_viz.ghost_spore:
            # Устанавливаем полупрозрачность
            base_color = self.color_manager.get_color('spore', 'default')
            try:
                prediction_viz.ghost_spore.color = (base_color.r, base_color.g, base_color.b, alpha)
            except:
                prediction_viz.ghost_spore.color = (0.6, 0.4, 0.9, alpha)

            # Обновляем позицию
            prediction_viz.update(final_position)

            # Безопасно регистрируем призрака в zoom_manager
            try:
                ghost_id = f'tree_ghost_{name_suffix}'
                prediction_viz.ghost_spore.id = ghost_id
                prediction_viz.ghost_spore._reg_id = ghost_id  # Для корректного удаления
                self.zoom_manager.register_object(prediction_viz.ghost_spore, ghost_id)
            except Exception as e:
                print(f"Ошибка регистрации призрака {name_suffix}: {e}")

        # Добавляем в список предсказаний через PreviewManager
        self.preview_manager.add_prediction_visualizer(prediction_viz)
        return prediction_viz

    def _create_ghost_link(self, parent_spore, child_spore, link_suffix, color_name):
        """Создает призрачный линк между двумя спорами."""
        try:
            from ..visual.link import Link

            ghost_link = Link(
                parent_spore=parent_spore,
                child_spore=child_spore,
                color_manager=self.color_manager,
                zoom_manager=self.zoom_manager,
                config=self.config
            )

            # Устанавливаем цвет линка
            ghost_link.color = self.color_manager.get_color('link', color_name)

            # Делаем линк полупрозрачным
            if hasattr(ghost_link, 'alpha'):
                ghost_link.alpha = 0.6

            # Обновляем геометрию (БЕЗ регистрации в zoom_manager для временных объектов)
            ghost_link.update_geometry()
            # link_id = f'ghost_link_{link_suffix}'
            # ghost_link._reg_id = link_id  # Для корректного удаления - НЕ НУЖНО для временных
            # self.zoom_manager.register_object(ghost_link, link_id)  # ← ЗАКОММЕНТИРОВАТЬ

            # Добавляем в список для очистки через PreviewManager
            self.preview_manager.add_prediction_link(ghost_link)

        except Exception as e:
            print(f"Ошибка создания призрачного линка {link_suffix}: {e}")


    


    def _destroy_preview(self) -> None:
        """Уничтожает превью спору, предсказания и их линки. Делегирует в PreviewManager."""
        self.preview_manager.clear_all()
    
 

    def clear_all(self) -> None:
        """Очищает все ресурсы созданные ManualSporeManager."""
        print("🧹 ManualSporeManager: очистка всех ресурсов...")
        
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
                link.parent = None    # Отвязываем от родителя
                
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
        self.creation_history.clear_all()
        
        self.created_links.clear()

    def destroy(self) -> None:
        """Очищает все ресурсы менеджера."""
        self.clear_all()  # Используем новый метод
        print("   ✓ Manual Spore Manager уничтожен")

    def delete_last_spore_group(self) -> bool:
        """
        Удаляет последнюю созданную группу спор и их линки.
        Делегирует вызов в CreationHistory.
        """
        return self.creation_history.delete_last_group()

    def create_spore_at_cursor(self) -> Optional[List[Spore]]:
        """Создает споры в позиции курсора. Делегирует в SporeCreator."""
        created_spores = self.spore_creator.create_spore_at_cursor(self.preview_manager.get_preview_spore())
        if created_spores:
            # Получаем все созданные линки из spore_creator
            all_links = self.spore_creator.created_links

            # Для спор: последние 4 линка, для деревьев: все линки после предыдущего count
            if self.spore_creator.get_creation_mode() == 'spores':
                created_links = all_links[-4:] if len(all_links) >= 4 else all_links
            else:  # tree mode
                # Для деревьев используем все недавно созданные линки
                created_links = all_links[max(0, len(all_links)-20):]  # Последние 20 линков

            self.creation_history.add_group(created_spores, created_links)
        return created_spores

    def toggle_creation_mode(self) -> None:
        """Переключает режим создания."""
        current_mode = self.spore_creator.get_creation_mode()
        if current_mode == 'spores':
            self.spore_creator.set_creation_mode('tree')
            self.preview_manager.set_creation_mode('tree')
            print("🌲 Режим: Создание деревьев")
        else:
            self.spore_creator.set_creation_mode('spores')
            self.preview_manager.set_creation_mode('spores')
            print("🌟 Режим: Создание спор")

    def set_tree_depth(self, depth: int) -> None:
        """Устанавливает глубину дерева."""
        self.spore_creator.set_tree_depth(depth)
        print(f"🌲 Глубина дерева: {self.spore_creator.get_tree_depth()}")

    @property
    def creation_mode(self) -> str:
        """Возвращает текущий режим создания."""
        return self.spore_creator.get_creation_mode()

    @property
    def tree_depth(self) -> int:
        """Возвращает текущую глубину дерева."""
        return self.spore_creator.get_tree_depth()

    @property
    def created_links(self) -> List[Link]:
        """Возвращает список созданных линков."""
        return self.spore_creator.created_links

    def update_ghost_tree_with_optimal_pairs(self):
        """Обновляет призрачное дерево оптимальными парами. Делегирует в TreeOptimizer."""
        if hasattr(self, 'current_ghost_tree'):
            self.tree_optimizer.update_ghost_tree_with_optimal_pairs(self.current_ghost_tree)

    def reset_ghost_tree_optimization(self):
        """Сбрасывает оптимизацию. Делегирует в TreeOptimizer."""
        self.tree_optimizer.reset_ghost_tree_optimization()

    def _debug_dump_tree(self, tag: str, tree_logic):
        """Диагностическая функция для отладки деревьев."""
        try:
            ch = getattr(tree_logic, 'children', [])
            gc = getattr(tree_logic, 'grandchildren', [])
            print(f"[{tag}] children={len(ch)} grandchildren={len(gc)}")
            if ch:
                dtc = [round(c.get('dt', float('nan')), 6) for c in ch]
                uc = [round(c.get('control', float('nan')), 6) for c in ch]
                print(f"[{tag}] dt_children={dtc}")
                print(f"[{tag}] u_children={uc}")
            if gc:
                dtg = [round(g.get('dt', float('nan')), 6) for g in gc]
                ug = [round(g.get('control', float('nan')), 6) for g in gc]
                print(f"[{tag}] dt_grandchildren={dtg}")
                print(f"[{tag}] u_grandchildren={ug}")
        except Exception as e:
            print(f"[{tag}] dump failed: {e}")




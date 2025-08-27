from ursina import held_keys, mouse
import time
import numpy as np
from typing import Optional

# Предварительное объявление классов для аннотаций типов
# Это позволяет избежать циклических импортов
from ..visual.scene_setup import SceneSetup
from ..managers.zoom_manager import ZoomManager
from ..managers.spore_manager import SporeManager
from ..managers.spawn_area_manager import SpawnAreaManager
from ..managers.param_manager import ParamManager
from ..visual.ui_setup import UI_setup

# Forward declaration для ManualSporeManager
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..managers.manual_spore_manager import ManualSporeManager
    from ..managers.angel_manager import AngelManager
    from ..visual.cost_visualizer import CostVisualizer
    from ..managers.dt_manager import DTManager
    
from ..utils.debug_output import always_print


class InputManager:
    """
    Централизованный класс для обработки всего пользовательского ввода.
    Принимает ссылки на все необходимые менеджеры и объекты сцены,
    и делегирует им команды в зависимости от нажатой клавиши.
    """
    def __init__(self, 
                 scene_setup: Optional[SceneSetup] = None, 
                 zoom_manager: Optional[ZoomManager] = None, 
                 spore_manager: Optional[SporeManager] = None, 
                 spawn_area_manager: Optional[SpawnAreaManager] = None, 
                 param_manager: Optional[ParamManager] = None, 
                 ui_setup: Optional[UI_setup] = None,
                 angel_manager: Optional['AngelManager'] = None,
                 cost_visualizer: Optional['CostVisualizer'] = None,
                 manual_spore_manager: Optional["ManualSporeManager"] = None,
                 dt_manager: Optional['DTManager'] = None):
        
        self.scene_setup: Optional[SceneSetup] = scene_setup
        self.zoom_manager: Optional[ZoomManager] = zoom_manager
        self.spore_manager: Optional[SporeManager] = spore_manager
        self.dt_manager: Optional['DTManager'] = dt_manager
        self.spawn_area_manager: Optional[SpawnAreaManager] = spawn_area_manager
        self.param_manager: Optional[ParamManager] = param_manager
        self.ui_setup: Optional[UI_setup] = ui_setup
        self.angel_manager: Optional['AngelManager'] = angel_manager
        self.cost_visualizer: Optional['CostVisualizer'] = cost_visualizer
        self.manual_spore_manager: Optional["ManualSporeManager"] = manual_spore_manager
        self.dt_manager: Optional['DTManager'] = dt_manager

        # Настройки для генерации спор по клавише 'f'
        self.f_key_down_time: float = 0
        self.long_press_threshold: float = 0.4
        self.spawn_interval: float = 0.1
        self.next_spawn_time: float = 0  # Время следующего спавна
        
        # v13_manual: отслеживание кликов мыши
        self.previous_mouse_left = False

    def update(self) -> None:
        """
        Этот метод должен вызываться каждый кадр для обработки непрерывного ввода.
        """
        # v13_manual: обработка кликов мыши для создания спор
        if self.manual_spore_manager:
            current_mouse_left = mouse.left
            # Обнаруживаем нажатие ЛКМ (переход с False на True)
            if current_mouse_left and not self.previous_mouse_left:
                created_spores = self.manual_spore_manager.create_spore_at_cursor()
                if created_spores:
                    print(f"   🖱️ ЛКМ: Создано {len(created_spores)} спор (1 родитель + 2 ребёнка + 2 линка)")
            self.previous_mouse_left = current_mouse_left
        
        # Логика для непрерывной генерации спор при удержании 'f'
        if held_keys['f']:  # type: ignore
            if self.f_key_down_time > 0:  # Проверяем, что нажатие было зафиксировано
                now = time.time()
                if now >= self.next_spawn_time:
                    if self.spore_manager:
                        self.spore_manager.generate_new_spore()
                    # Устанавливаем время для следующей генерации
                    self.next_spawn_time = now + self.spawn_interval
        else:
            # Сбрасываем таймер если клавиша не нажата
            self.f_key_down_time = 0

    def handle_input(self, key: str) -> None:
        """
        Основной метод-обработчик. Вызывается из главного цикла приложения.
        """
        # Отладка: выводим полученную клавишу и состояние курсора
        # print(f"[InputManager] handle_input called with key: '{key}'. Cursor locked: {self.scene_setup.cursor_locked if self.scene_setup else 'N/A'}")

        # Базовое управление (выход) было перенесено на глобальный уровень
        # в main_demo.py, чтобы работать независимо от заморозки ввода.
        # if key == 'q' or key == 'escape':
        #     application.quit()
        #     return

        # Остальные команды работают только когда курсор свободен
        # if self.scene_setup and self.scene_setup.cursor_locked:
        #     # Отладка: выводим, почему мы прерываем выполнение
        #     print(f"[InputManager] Aborting: Cursor is locked.")
        #     return
        
        # Этот принт был в файле, но я его оставлю для дополнительной отладки
        # print(f"[InputManager] Passed lock check. Processing key '{key}'...")
        # # 1. UI команды (самый высокий приоритет)
        # if self.ui_setup:
        #     # Отладка: что возвращает обработчик UI?
        #     print(f"[InputManager] Calling ui_setup.handle_demo_commands('{key}')...")
        #     processed = self.ui_setup.handle_demo_commands(key)
        #     print(f"[InputManager] ...ui_setup.handle_demo_commands returned: {processed}")
        #     # handle_demo_commands вернет True, если команда была обработана
        #     if processed:
        #         return

        # 2. Управление спорами
        if key == 'f':
            if self.spore_manager:
                self.spore_manager.generate_new_spore()
            # Запускаем таймеры для возможной непрерывной генерации
            now = time.time()
            self.f_key_down_time = now
            self.next_spawn_time = now + self.long_press_threshold
            return
        
        if key == 'g':
            if self.spore_manager:
                self.spore_manager.activate_random_candidate()
            return
        
        if key == 'v':
            if self.spore_manager:
                self.spore_manager.evolve_all_candidates_to_completion()
            return
        
        if key == 'p':
            # Применить оптимальные пары к призрачному дереву под курсором
            if self.manual_spore_manager:
                self._apply_optimal_pairs_to_ghost_tree()
            return
        
        if key == 'u':
            if self.scene_setup and hasattr(self.scene_setup, 'frame'):
                self.scene_setup.frame.toggle_visibility()  # type: ignore
            return
            
        # 3. Управление масштабированием
        if self.zoom_manager:
                # Проверяем нажат ли Ctrl
            ctrl_pressed = held_keys['left control'] or held_keys['right control']  # type: ignore
            
            if ctrl_pressed:
                # Ctrl + колесико = управление dt
                if key == 'scroll up' and self.dt_manager:
                    self.dt_manager.increase_dt()
                    return
                elif key == 'scroll down' and self.dt_manager:
                    self.dt_manager.decrease_dt()
                    return
            else:
                # Обычное колесико = зум
                if key == 'e' or key == 'scroll up':
                    self.zoom_manager.zoom_in()
                elif key == 't' or key == 'scroll down':
                    self.zoom_manager.zoom_out()
        
        # Остальные команды зума
            if key == 'r': 
                self.zoom_manager.reset_zoom()
            elif key == '1': 
                self.zoom_manager.increase_spores_scale()
            elif key == '2': 
                self.zoom_manager.decrease_spores_scale()

        if key == 'm':  # Reset dt to original
            if self.dt_manager:
                self.dt_manager.reset_dt()
            return

        # 4. Показ статистики dt
        if key == 'j':  # Show dt info
            if self.dt_manager:
                self.dt_manager.print_stats()
            return

        
        # 4. Очистка объектов (v13_manual)
        if  held_keys['c'] and self.spore_manager and held_keys['left control']:  # type: ignore
            print("🧹 Клавиша C: Полная очистка всех спор и объектов")
            
            # 🆕 ДИАГНОСТИКА: Проверяем есть ли manual_spore_manager
            if self.manual_spore_manager:
                print(f"   🔍 ManualSporeManager найден: {type(self.manual_spore_manager)}")
                if hasattr(self.manual_spore_manager, 'clear_all'):
                    print(f"   🔍 Метод clear_all найден")
                    print(f"   📊 created_links до очистки: {len(self.manual_spore_manager.created_links)}")
                    self.manual_spore_manager.clear_all()  
                else:
                    print(f"   ❌ Метод clear_all НЕ найден!")
            else:
                print(f"   ❌ ManualSporeManager НЕ найден!")
            
            self.spore_manager.clear_all_manual()
            
        # 5. Управление параметром
        # if self.param_manager:
        #     if key == 'z': 
        #         self.param_manager.increase()
        #     elif key == 'x': 
        #         self.param_manager.decrease()


        if key == 'z' or key == 'backspace':
            if self.manual_spore_manager:
                print("🗑️ Клавиша удаления: Удаление последней группы спор")
                success = self.manual_spore_manager.delete_last_spore_group()
                if success:
                    print("   ✅ Последняя группа спор успешно удалена")
                else:
                    print("   ❌ Не удалось удалить группу (возможно, нет групп для удаления)")
            else:
                print("   ❌ ManualSporeManager не найден!")
            return

        # 5. Управление областью спавна
        if self.spawn_area_manager:
            if key == '3' or key == 'arrow_up': 
                self.spawn_area_manager.decrease_eccentricity()
            elif key == '4' or key == 'arrow_down': 
                self.spawn_area_manager.increase_eccentricity()
        
        # 6. Управление радиусом кандидатов
        if self.spore_manager:
            if key == '5':
                self.spore_manager.adjust_min_radius(1/1.2)  # Уменьшить радиус (÷1.2)
            elif key == '6':
                self.spore_manager.adjust_min_radius(1.2)    # Увеличить радиус (×1.2)
        
        # 7. Переключение ангелов
        if key == 'y':
            if self.angel_manager:
                self.angel_manager.toggle_angels()
            else:
                always_print("⚠️ Angel Manager не найден")
        
        # Команды деревьев
        if key == 'k':
            if self.manual_spore_manager and hasattr(self.manual_spore_manager, 'toggle_creation_mode'):
                self.manual_spore_manager.toggle_creation_mode()
            return

        # Глубина дерева (только в режиме дерева)
        if self.manual_spore_manager and hasattr(self.manual_spore_manager, 'creation_mode'):
            if self.manual_spore_manager.creation_mode == 'tree':
                if key == '7' and hasattr(self.manual_spore_manager, 'set_tree_depth'):
                    self.manual_spore_manager.set_tree_depth(1)
                    return
                if key == '8' and hasattr(self.manual_spore_manager, 'set_tree_depth'):
                    self.manual_spore_manager.set_tree_depth(2)
                    return

        # Оптимизация дерева
        if key == 'o':
            if self.manual_spore_manager and hasattr(self.manual_spore_manager, 'optimize_tree'):
                self.manual_spore_manager.optimize_tree()  # Если добавите этот метод
            return

    def _apply_optimal_pairs_to_ghost_tree(self) -> None:
        """
        Находит оптимальные пары для призрачного дерева под курсором
        и подставляет соответствующие dt в ghost_tree_dt_vector
        """
        try:
            # Проверяем что мы в режиме дерева глубины 2
            if not self.manual_spore_manager:
                print(f"❌ ManualSporeManager не найден")
                return
                
            if self.manual_spore_manager.creation_mode != 'tree' or self.manual_spore_manager.tree_depth != 2:
                print(f"❌ Оптимизация пар доступна только для деревьев глубины 2")
                print(f"   Текущий режим: {self.manual_spore_manager.creation_mode}")
                print(f"   Текущая глубина: {self.manual_spore_manager.tree_depth}")
                return
            
            # Получаем позицию курсора
            mouse_pos = self.manual_spore_manager.get_mouse_world_position()
            if mouse_pos is None:
                print("❌ Не удалось получить позицию курсора")
                return

            # Конвертируем в numpy array
            cursor_position_2d = np.array([mouse_pos[0], mouse_pos[1]])
            
            print(f"🎯 Поиск оптимальных пар для позиции {cursor_position_2d}")
            
            # Импортируем нужные классы
            from ..logic.tree.spore_tree import SporeTree
            from ..logic.tree.spore_tree_config import SporeTreeConfig
            from ..logic.tree.pairs.find_optimal_pairs import find_optimal_pairs
            
            # Получаем текущий dt из системы
            dt = self.dt_manager.get_dt() if self.dt_manager else 0.05
            
            # Создаем временное дерево для поиска пар
            tree_config = SporeTreeConfig(
                initial_position=cursor_position_2d,
                dt_base=dt,
                dt_grandchildren_factor=0.2,
                show_debug=False
            )
            
            temp_tree = SporeTree(
                pendulum=self.manual_spore_manager.deps.pendulum,
                config=tree_config,
                auto_create=True  # Создает полное дерево автоматически
            )
            
            print(f"🌲 Временное дерево создано: {len(temp_tree.children)} детей, {len(temp_tree.grandchildren)} внуков")
            
            # Ищем оптимальные пары
            pairs = find_optimal_pairs(temp_tree, show=True)
            
            if pairs is None:
                print(f"❌ Не удалось найти пары")
                return
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: должно быть ровно 4 пары
            if len(pairs) != 4:
                raise Exception(f"ОШИБКА: Найдено {len(pairs)} пар, а должно быть ровно 4!")
            
            print(f"✅ Найдено {len(pairs)} оптимальных пар")
            
            # Извлекаем dt из пар
            # Инициализируем массивы исходными значениями
            dt_children = np.array([child['dt'] for child in temp_tree.children])
            dt_grandchildren = np.array([gc['dt'] for gc in temp_tree.grandchildren])
            
            print(f"📊 Исходные dt детей: {dt_children}")
            print(f"📊 Исходные dt внуков: {dt_grandchildren}")
            
            # Обновляем dt внуков согласно найденным парам
            for pair_idx, (gc_i, gc_j, meeting_info) in enumerate(pairs):
                # Извлекаем оптимальные времена (ВАЖНО: сохраняем знаки!)
                optimal_dt_i = meeting_info['time_gc']      # время для gc_i
                optimal_dt_j = meeting_info['time_partner'] # время для gc_j
                
                # Обновляем массив dt_grandchildren
                dt_grandchildren[gc_i] = optimal_dt_i
                dt_grandchildren[gc_j] = optimal_dt_j
                
                print(f"   Пара {pair_idx+1}: gc_{gc_i} ↔ gc_{gc_j}")
                print(f"     gc_{gc_i}: {temp_tree.grandchildren[gc_i]['dt']:+.6f} → {optimal_dt_i:+.6f}")
                print(f"     gc_{gc_j}: {temp_tree.grandchildren[gc_j]['dt']:+.6f} → {optimal_dt_j:+.6f}")
                print(f"     Расстояние встречи: {meeting_info['distance']:.6f}")
            
            # Формируем финальный dt_vector из 12 элементов
            # Первые 4 - dt детей, следующие 8 - dt внуков
            dt_vector = np.concatenate([dt_children, dt_grandchildren])
            
            print(f"🎯 Сформированный dt_vector:")
            print(f"   dt_children (0:4): {dt_vector[:4]}")  
            print(f"   dt_grandchildren (4:12): {dt_vector[4:12]}")
            
            # Подставляем в призрачное дерево
            self.manual_spore_manager.ghost_tree_dt_vector = dt_vector
            
            # ВАЖНО: Принудительно обновляем предсказания чтобы показать спаренное дерево
            # Принудительно обновляем призрачное дерево с новым dt_vector
            # Очищаем старые предсказания и пересоздаем с новым dt_vector
            self.manual_spore_manager.prediction_manager.clear_predictions()
            self.manual_spore_manager._update_predictions()

            print(f"🔄 Призрачное дерево пересоздано с новыми dt")
            
            print(f"🌲 Призрачное дерево обновлено со спаренными dt!")
            
        except Exception as e:
            print(f"❌ Ошибка применения пар: {e}")
            import traceback
            traceback.print_exc()
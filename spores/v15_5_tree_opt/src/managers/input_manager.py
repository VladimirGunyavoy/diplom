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
from ..logic.tree import run_area_optimization

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

        print(f"[IM] constructed, dt_manager id={id(self.dt_manager) if self.dt_manager else None}")

        # 🔍 Флаг для включения детальной отладки призрачного дерева
        self.debug_ghost_tree = False

        # Настройки для генерации спор по клавише 'f'
        self.f_key_down_time: float = 0
        self.long_press_threshold: float = 0.4
        self.spawn_interval: float = 0.1
        self.next_spawn_time: float = 0  # Время следующего спавна
        
        # v13_manual: отслеживание кликов мыши
        self.previous_mouse_left = False

        # Подписка на изменения dt (чтобы призраки и линки всегда реагировали)
        if self.dt_manager and hasattr(self.dt_manager, "subscribe_on_change"):
            self.dt_manager.subscribe_on_change(self._on_dt_changed)
            print("[IM] subscribed to DTManager.on_change")

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
            ctrl_pressed = held_keys['left control'] or held_keys['right control']  # type: ignore

            if ctrl_pressed:
                # Ctrl + колесико = управление dt
                changed = False
                if (key == 'e' or key == 'scroll up') and self.dt_manager:
                    self.dt_manager.increase_dt()
                    changed = True
                elif (key == 't' or key == 'scroll down') and self.dt_manager:
                    self.dt_manager.decrease_dt()
                    changed = True

                if changed:
                    # 🔧 форсируем пересчёт даже если подписка потерялась
                    if hasattr(self, "_on_dt_changed"):
                        print("[IM] Ctrl+Wheel: forcing _on_dt_changed()")
                        try:
                            self._on_dt_changed()
                        except Exception as ex:
                            print(f"[IM] Ctrl+Wheel: _on_dt_changed() error: {ex}")
                    # Не даём провалиться в обычный зум
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

        if key == '[':  # Reset all dt to standard mode
            # 1. Сбрасываем общий dt через dt_manager (как клавиша M)
            if self.dt_manager:
                self.dt_manager.reset_dt()
            
            # 2. Сбрасываем ghost_tree_dt_vector к стандартным значениям
            if self.manual_spore_manager:
                # Получаем текущий dt после сброса
                current_dt = self.dt_manager.get_dt() if self.dt_manager else 0.001
                
                # Получаем фактор для внуков из конфигурации
                factor = 0.05  # дефолтное значение
                if self.manual_spore_manager and hasattr(self.manual_spore_manager, 'deps'):
                    factor = self.manual_spore_manager.deps.config.get('tree', {}).get('dt_grandchildren_factor', 0.05)
                
                # Формируем стандартный dt_vector: 4 детей + 8 внуков
                dt_children = np.array([+current_dt, -current_dt, +current_dt, -current_dt], dtype=float)
                base_gc = current_dt * factor
                dt_grandchildren = np.array([
                    +base_gc, -base_gc, +base_gc, -base_gc,
                    +base_gc, -base_gc, +base_gc, -base_gc
                ], dtype=float)
                
                standard_dt_vector = np.concatenate([dt_children, dt_grandchildren])
                
                # Устанавливаем стандартный вектор
                self.manual_spore_manager.ghost_tree_dt_vector = standard_dt_vector
                
                # Обновляем предсказания
                if hasattr(self.manual_spore_manager, 'prediction_manager'):
                    self.manual_spore_manager.prediction_manager.clear_predictions()
                    self.manual_spore_manager._update_predictions()
                
                print(f"🔄 Все dt сброшены к стандартным значениям:")
                print(f"   📊 Общий dt: {current_dt}")
                print(f"   📊 dt детей: {dt_children}")
                print(f"   📊 dt внуков (factor={factor}): {dt_grandchildren}")
                print(f"   📊 Полный dt_vector: {standard_dt_vector}")
            
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
        if key == 'o' or key == 'O':
            print(f"[IM][O] Клавиша O нажата! key='{key}'")
            try:
                print("[IM][O] Запуск оптимизации площади...")

                # ==== Достаём зависимости ====
                # pendulum находится в deps
                pendulum = getattr(self.manual_spore_manager, "deps", None)
                if pendulum:
                    pendulum = getattr(pendulum, "pendulum", None)

                # dt-manager
                dt_manager = self.dt_manager

                print(f"[IM][O] Диагностика зависимостей:")
                print(f"   pendulum: {pendulum is not None} ({type(pendulum) if pendulum else 'None'})")
                print(f"   dt_manager: {dt_manager is not None} ({type(dt_manager) if dt_manager else 'None'})")

                if pendulum is None:
                    raise RuntimeError("Не найден объект pendulum в manual_spore_manager.deps.pendulum")
                if dt_manager is None:
                    raise RuntimeError("Не найден dt_manager")

                # ==== Создаём временное дерево и пары ====
                print("[IM][O] Создание временного дерева для оптимизации...")
                
                # Получаем позицию курсора
                mouse_pos = self.manual_spore_manager.get_mouse_world_position()
                if mouse_pos is None:
                    raise RuntimeError("Не удалось получить позицию курсора")
                
                cursor_position_2d = np.array([mouse_pos[0], mouse_pos[1]])
                print(f"[IM][O] Позиция курсора: {cursor_position_2d}")
                
                # Импортируем нужные классы
                from ..logic.tree.spore_tree import SporeTree
                from ..logic.tree.spore_tree_config import SporeTreeConfig
                from ..logic.tree.pairs.find_optimal_pairs import find_optimal_pairs
                
                # Загружаем конфигурацию спаривания
                from ..logic.tree.tree_area_bridge import _load_pairing_config
                pairing_config = _load_pairing_config()
                
                # Получаем текущий dt из системы
                dt = dt_manager.get_dt() if dt_manager else 0.05
                
                # Создаем временное дерево для поиска пар
                tree_config = SporeTreeConfig(
                    initial_position=cursor_position_2d,
                    dt_base=dt,
                    dt_grandchildren_factor=pairing_config.get('dt_grandchildren_factor', 0.2),
                    show_debug=pairing_config.get('show_debug', True)
                )
                
                temp_tree = SporeTree(
                    pendulum=pendulum,
                    config=tree_config,
                    auto_create=True  # Создает полное дерево автоматически
                )
                
                print(f"[IM][O] Временное дерево создано: {len(temp_tree.children)} детей, {len(temp_tree.grandchildren)} внуков")
                
                # Ищем оптимальные пары
                pairs = find_optimal_pairs(temp_tree, show=True)
                
                if pairs is None:
                    raise RuntimeError("Не удалось найти пары")
                
                print(f"[IM][O] Найдено {len(pairs)} оптимальных пар")

                # ==== Вызов оптимизации ====
                # Параметры загружаются из config/json/config.json
                result = run_area_optimization(
                    tree=temp_tree,
                    pairs=pairs,
                    pendulum=pendulum,
                    dt_manager=dt_manager,
                    # dt_bounds, optimization_method, max_iterations загружаются из конфига
                )

                # ==== Красивые логи (научная нотация) ====
                if result is None:
                    print(f"[IM][O] ❌ Оптимизация вернула None - возможно, произошла ошибка")
                    return
                    
                try:
                    start_area = result.get("start_area", result.get("initial_area", None))
                    best_area  = result.get("best_area",  result.get("optimized_area", None))
                    if start_area is not None and best_area is not None:
                        delta = best_area - start_area
                        rel = (delta / max(abs(start_area), 1e-12)) * 100.0
                        print(f"[IM][O] Готово. Площадь: {start_area:.6e} → {best_area:.6e}  "
                              f"(Δ={delta:.3e}, {rel:+.2f}%)")
                    else:
                        print(f"[IM][O] Оптимизация завершена. result-ключи: {list(result.keys())}")
                        
                    # Дополнительная информация о результатах
                    success = result.get("success", False)
                    if success:
                        print(f"[IM][O] ✅ Оптимизация успешно завершена!")
                        
                        # Показываем улучшение
                        improvement = result.get("improvement", None)
                        improvement_percent = result.get("improvement_percent", None)
                        if improvement is not None:
                            print(f"[IM][O] 🎯 Улучшение площади: {improvement:.6e}")
                        if improvement_percent is not None:
                            print(f"[IM][O] 📊 Процентное улучшение: {improvement_percent:+.2f}%")
                        
                        # Показываем оптимизированные dt
                        optimized_dt_vector = result.get("optimized_dt_vector", None)
                        if optimized_dt_vector is not None:
                            print(f"[IM][O] ⏱️  Оптимизированные dt:")
                            print(f"   Дети: {optimized_dt_vector[:4]}")
                            print(f"   Внуки: {optimized_dt_vector[4:12]}")
                        
                        # Показываем статистику констрейнтов
                        constraint_violations = result.get("constraint_violations", {})
                        if constraint_violations:
                            violations = constraint_violations.get("violations", [])
                            if violations:
                                max_violation = max(violations)
                                print(f"[IM][O] ⚠️  Максимальное нарушение констрейнта: {max_violation:.6e}")
                            else:
                                print(f"[IM][O] ✅ Все констрейнты соблюдены")
                        
                        # Показываем дистанции по парам
                        constraint_violations = result.get("constraint_violations", {})
                        if constraint_violations:
                            print(f"[IM][O] 📏 Дистанции по парам:")
                            pair_distances = []
                            for i in range(len(constraint_violations)):
                                if isinstance(constraint_violations.get(i), dict):
                                    distance = constraint_violations[i].get('distance', None)
                                    if distance is not None:
                                        pair_distances.append(distance)
                                        print(f"   Пара {i+1}: {distance:.6e}")
                            
                            if pair_distances:
                                min_distance = min(pair_distances)
                                max_distance = max(pair_distances)
                                print(f"[IM][O] 📊 Минимальная дистанция: {min_distance:.6e}")
                                print(f"[IM][O] 📊 Максимальная дистанция: {max_distance:.6e}")
                                print(f"[IM][O] 📊 Средняя дистанция: {sum(pair_distances)/len(pair_distances):.6e}")
                            else:
                                print(f"[IM][O] 📏 Дистанции по парам: не найдены в constraint_violations")
                        else:
                            print(f"[IM][O] 📏 Дистанции по парам: constraint_violations пуст")
                        
                                                # 🔧 ПРИМЕНЯЕМ ОПТИМИЗИРОВАННЫЕ DT К ПРИЗРАЧНОМУ ДЕРЕВУ
                        optimized_dt_vector = result.get("optimized_dt_vector", None)
                        if optimized_dt_vector is not None and self.manual_spore_manager:
                            # 🔍 ИСПРАВЛЕНИЕ ЗНАКОВ: используем тот же подход, что и в методе 'P'
                            # Дети должны иметь те же знаки, что и исходные dt детей
                            dt_children_original = np.array([child['dt'] for child in temp_tree.children])
                            dt_grandchildren_original = np.array([gc['dt'] for gc in temp_tree.grandchildren])
                            
                            # Исправляем знаки детей (должны быть как в исходном дереве)
                            optimized_dt_vector[:4] = np.sign(dt_children_original) * np.abs(optimized_dt_vector[:4])
                            
                            # Исправляем знаки внуков (должны быть как в исходном дереве)
                            optimized_dt_vector[4:12] = np.sign(dt_grandchildren_original) * np.abs(optimized_dt_vector[4:12])
                            
                            # Применяем dt_vector к призрачному дереву (как в методе 'P')
                            self.manual_spore_manager.ghost_tree_dt_vector = optimized_dt_vector
                            
                            # Обновляем baseline для масштабирования (как в методе 'P')
                            if self.dt_manager:
                                self.manual_spore_manager.ghost_dt_baseline = self.dt_manager.get_dt()
                            
                            # Принудительно обновляем призрачное дерево (как в методе 'P')
                            if hasattr(self.manual_spore_manager, 'prediction_manager'):
                                self.manual_spore_manager.prediction_manager.clear_predictions()
                                self.manual_spore_manager._update_predictions()
                                print(f"[IM][O] ✅ Призрачное дерево обновлено с оптимизированными dt!")
                            else:
                                print(f"[IM][O] ⚠️  PredictionManager не найден, не удалось обновить призрачное дерево")
                        else:
                            print(f"[IM][O] ⚠️  Оптимизированные dt не найдены или ManualSporeManager недоступен")
                    else:
                        print(f"[IM][O] ❌ Оптимизация не удалась")
                        
                except Exception as e:
                    print(f"[IM][O] Оптимизация завершена, но не смог распечатать метрики: {e}")

            except Exception as e:
                print(f"[IM][O] Ошибка оптимизации площади: {e}")
            return
        
        # 🔍 Переключение детальной отладки призрачного дерева
        if key == 'h':
            self.debug_ghost_tree = not self.debug_ghost_tree
            status = "ВКЛЮЧЕНА" if self.debug_ghost_tree else "ОТКЛЮЧЕНА"
            print(f"🔍 Детальная отладка призрачного дерева: {status}")
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
            
            # 🔍 ДИАГНОСТИКА ДО СПАРИВАНИЯ: Показываем текущее состояние дерева
            if self.debug_ghost_tree:
                print(f"\n🔍 ДИАГНОСТИКА ДО СПАРИВАНИЯ:")
                
                # Получаем текущее призрачное дерево ДО изменений
                if hasattr(self.manual_spore_manager, 'prediction_manager') and self.manual_spore_manager.prediction_manager:
                    pred_manager = self.manual_spore_manager.prediction_manager
                    
                    # Показываем текущий dt_vector
                    if hasattr(self.manual_spore_manager, 'ghost_tree_dt_vector') and self.manual_spore_manager.ghost_tree_dt_vector is not None:
                        current_dt_vector = self.manual_spore_manager.ghost_tree_dt_vector
                        print(f"   📊 Текущий ghost_tree_dt_vector:")
                        print(f"      Дети (0:4): {current_dt_vector[:4]}")
                        print(f"      Внуки (4:12): {current_dt_vector[4:12]}")
                        
                        # 🔍 Сохраняем dt детей ДО для сравнения
                        self._children_dt_before = current_dt_vector[:4].copy()
                        print(f"   💾 Сохранили dt детей ДО: {self._children_dt_before}")
                    else:
                        print(f"   📊 Текущий ghost_tree_dt_vector: НЕ УСТАНОВЛЕН")
                        self._children_dt_before = None
                    
                    # Показываем количество призраков
                    if hasattr(pred_manager, 'prediction_visualizers'):
                        ghost_count = len([v for v in pred_manager.prediction_visualizers if v.ghost_spore])
                        print(f"   👻 Призрачных спор: {ghost_count}")
                    
                    # Показываем количество линков
                    if hasattr(pred_manager, 'prediction_links'):
                        print(f"   🔗 Призрачных линков: {len(pred_manager.prediction_links)}")
                    
                    # Показываем позиции всех призраков
                    if hasattr(pred_manager, 'prediction_visualizers'):
                        print(f"   📍 Позиции призраков ДО:")
                        for i, viz in enumerate(pred_manager.prediction_visualizers):
                            if viz.ghost_spore:
                                pos = (viz.ghost_spore.x, viz.ghost_spore.z)
                                print(f"      Призрак {i}: {pos}")
                        
                        # 🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА ВИЗУАЛЬНЫХ ОБЪЕКТОВ ДО
                        print(f"   🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА ВИЗУАЛЬНЫХ ОБЪЕКТОВ ДО:")
                        for i, viz in enumerate(pred_manager.prediction_visualizers):
                            if viz.ghost_spore:
                                print(f"      Призрак {i}:")
                                print(f"         ID: {viz.id}")
                                print(f"         Позиция: ({viz.ghost_spore.x:.6f}, {viz.ghost_spore.z:.6f})")
                                print(f"         real_position: {viz.ghost_spore.real_position}")
                                print(f"         enabled: {viz.ghost_spore.enabled}")
                                if hasattr(viz.ghost_spore, 'logic'):
                                    print(f"         logic.position_2d: {viz.ghost_spore.logic.position_2d}")
                                print(f"         Тип: {type(viz.ghost_spore)}")
                else:
                    print(f"   ❌ PredictionManager не найден")
                
                print(f"🔍 ДИАГНОСТИКА ДО СПАРИВАНИЯ ЗАВЕРШЕНА\n")
            
            # Импортируем нужные классы
            from ..logic.tree.spore_tree import SporeTree
            from ..logic.tree.spore_tree_config import SporeTreeConfig
            from ..logic.tree.pairs.find_optimal_pairs import find_optimal_pairs
            
            # Загружаем конфигурацию спаривания
            from ..logic.tree.tree_area_bridge import _load_pairing_config
            pairing_config = _load_pairing_config()
            
            # Получаем текущий dt из системы
            dt = self.dt_manager.get_dt() if self.dt_manager else 0.05
            
            # Создаем временное дерево для поиска пар
            tree_config = SporeTreeConfig(
                initial_position=cursor_position_2d,
                dt_base=dt,
                dt_grandchildren_factor=pairing_config.get('dt_grandchildren_factor', 0.2),
                show_debug=pairing_config.get('show_debug', True)
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
            
            # 🔍 ВАЖНО: Дети НЕ ОПТИМИЗИРУЮТСЯ - их dt остаются стандартными!
            print(f"⚠️  ВНИМАНИЕ: dt детей НЕ изменяются алгоритмом спаривания!")
            print(f"   Дети используют стандартные dt: {dt_children}")
            print(f"   Только внуки получают оптимизированные dt из пар")
            
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

            # --- зафиксируем baseline для последующего масштабирования колесиком
            if self.manual_spore_manager is not None and self.dt_manager is not None:
                try:
                    cur_dt = float(self.dt_manager.get_dt())
                except Exception:
                    # fallback: возьмём из конфига, если есть
                    cur_dt = float(getattr(self.dt_manager, "current_dt", 0.1))
                self.manual_spore_manager.ghost_dt_baseline = cur_dt
                print(f"[IM] ghost_dt_baseline set to {cur_dt:.6f} after pairing")

            # Запомним базовый dt на момент вычисления пар — нужен для масштабирования
            if self.dt_manager:
                self.manual_spore_manager.ghost_dt_baseline = self.dt_manager.get_dt()
            
            # ВАЖНО: Принудительно обновляем предсказания чтобы показать спаренное дерево
            # Принудительно обновляем призрачное дерево с новым dt_vector
            # Очищаем старые предсказания и пересоздаем с новым dt_vector
            
            if self.debug_ghost_tree:
                print(f"🔄 Начинаем обновление призрачного дерева...")
                print(f"   📊 ghost_tree_dt_vector установлен: {dt_vector is not None}")
                print(f"   📊 Длина dt_vector: {len(dt_vector) if dt_vector is not None else 'None'}")
                
                # Передаем флаг отладки в prediction_manager
                self.manual_spore_manager.prediction_manager.debug_ghost_tree = True
                
                self.manual_spore_manager.prediction_manager.clear_predictions()
                print(f"   🧹 Предсказания очищены")
                
                self.manual_spore_manager._update_predictions()
                print(f"   🔄 _update_predictions() вызван")
                
                # Отключаем отладку после использования
                self.manual_spore_manager.prediction_manager.debug_ghost_tree = False
            else:
                self.manual_spore_manager.prediction_manager.clear_predictions()
                self.manual_spore_manager._update_predictions()

            print(f"🔄 Призрачное дерево пересоздано с новыми dt")
            
            print(f"✅ Призрачное дерево обновлено со спаренными dt!")
            
            # --- гарантируем подписку на изменение dt после спаривания
            if self.dt_manager and hasattr(self.dt_manager, "subscribe_on_change"):
                try:
                    self.dt_manager.subscribe_on_change(self._on_dt_changed)
                    print(f"[IM] re-subscribed to DTManager.on_change after pairing (DT id={id(self.dt_manager)})")
                    # Диагностика: покажем подписчиков
                    if hasattr(self.dt_manager, "debug_subscribers"):
                        self.dt_manager.debug_subscribers()
                except Exception as ex:
                    print(f"[IM] failed to re-subscribe after pairing: {ex}")
            
            # Сразу привести всё в соответствие текущему dt и лимитам длины
            if self.dt_manager and hasattr(self, "_on_dt_changed"):
                self._on_dt_changed()
            
            # Жёсткая синхронизация лимитов длины через PredictionManager
            if self.manual_spore_manager and hasattr(self.manual_spore_manager, "prediction_manager"):
                pm = self.manual_spore_manager.prediction_manager
                if hasattr(pm, "update_links_max_length"):
                    max_len = self.dt_manager.get_max_link_length() if self.dt_manager else None
                    print(f"[IM] Жёсткая синхронизация: applying max_len={max_len} to {len(getattr(pm,'prediction_links',[]))} prediction_links")
                    pm.update_links_max_length(max_len)
            
            # 🔍 ДИАГНОСТИКА ПОСЛЕ СПАРИВАНИЯ: Показываем состояние обновленного дерева
            if self.debug_ghost_tree:
                print(f"\n🔍 ДИАГНОСТИКА ПОСЛЕ СПАРИВАНИЯ:")
                
                # Получаем текущее призрачное дерево
                if hasattr(self.manual_spore_manager, 'prediction_manager') and self.manual_spore_manager.prediction_manager:
                    pred_manager = self.manual_spore_manager.prediction_manager
                    
                    # Показываем dt_vector
                    if hasattr(self.manual_spore_manager, 'ghost_tree_dt_vector') and self.manual_spore_manager.ghost_tree_dt_vector is not None:
                        dt_vector = self.manual_spore_manager.ghost_tree_dt_vector
                        print(f"   📊 ghost_tree_dt_vector:")
                        print(f"      Дети (0:4): {dt_vector[:4]} (НЕ ИЗМЕНЕНЫ - стандартные)")
                        print(f"      Внуки (4:12): {dt_vector[4:12]} (ОПТИМИЗИРОВАНЫ)")
                        
                        # 🔍 ПОДТВЕРЖДЕНИЕ: Дети действительно не изменились
                        print(f"   🔍 АНАЛИЗ dt детей:")
                        for i, dt in enumerate(dt_vector[:4]):
                            print(f"      Ребенок {i}: dt = {dt:+.6f} (стандартный)")
                        
                        # 🔍 СРАВНЕНИЕ dt детей ДО и ПОСЛЕ
                        if hasattr(self, '_children_dt_before') and self._children_dt_before is not None:
                            print(f"   🔍 СРАВНЕНИЕ dt детей:")
                            for i, (dt_before, dt_after) in enumerate(zip(self._children_dt_before, dt_vector[:4])):
                                if abs(dt_before - dt_after) < 1e-10:
                                    print(f"      Ребенок {i}: dt = {dt_before:+.6f} → {dt_after:+.6f} ✅ НЕ ИЗМЕНИЛСЯ")
                                else:
                                    print(f"      Ребенок {i}: dt = {dt_before:+.6f} → {dt_after:+.6f} 🔄 ИЗМЕНИЛСЯ!")
                        else:
                            print(f"   🔍 СРАВНЕНИЕ dt детей: Нет данных ДО для сравнения")
                    
                    # Показываем количество призраков
                    if hasattr(pred_manager, 'prediction_visualizers'):
                        ghost_count = len([v for v in pred_manager.prediction_visualizers if v.ghost_spore])
                        print(f"   👻 Призрачных спор: {ghost_count}")
                    
                    # Показываем количество линков
                    if hasattr(pred_manager, 'prediction_links'):
                        print(f"   🔗 Призрачных линков: {len(pred_manager.prediction_links)}")
                    
                    # Показываем позиции всех призраков
                    if hasattr(pred_manager, 'prediction_visualizers'):
                        print(f"   📍 Позиции призраков:")
                        for i, viz in enumerate(pred_manager.prediction_visualizers):
                            if viz.ghost_spore:
                                pos = (viz.ghost_spore.x, viz.ghost_spore.z)
                                print(f"      Призрак {i}: {pos}")
                        
                        # 🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА ВИЗУАЛЬНЫХ ОБЪЕКТОВ
                        print(f"   🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА ВИЗУАЛЬНЫХ ОБЪЕКТОВ:")
                        for i, viz in enumerate(pred_manager.prediction_visualizers):
                            if viz.ghost_spore:
                                print(f"      Призрак {i}:")
                                print(f"         ID: {viz.id}")
                                print(f"         Позиция: ({viz.ghost_spore.x:.6f}, {viz.ghost_spore.z:.6f})")
                                print(f"         real_position: {viz.ghost_spore.real_position}")
                                print(f"         enabled: {viz.ghost_spore.enabled}")
                                if hasattr(viz.ghost_spore, 'logic'):
                                    print(f"         logic.position_2d: {viz.ghost_spore.logic.position_2d}")
                                print(f"         Тип: {type(viz.ghost_spore)}")
                else:
                    print(f"   ❌ PredictionManager не найден")
                
                print(f"🔍 ДИАГНОСТИКА ЗАВЕРШЕНА\n")
            
        except Exception as e:
            print(f"❌ Ошибка применения пар: {e}")
            import traceback
            traceback.print_exc()

    def _on_dt_changed(self):
        """
        Глобальная реакция на изменение dt:
        - СНАЧАЛА масштабируем ghost_tree_dt_vector (если он есть),
        - потом пересобираем призрачное дерево,
        - применяем лимит длины ко всем линкам.
        """
        try:
            # 0) СНАЧАЛА: масштабируем dt-вектор спаренного дерева под новый dt
            if (self.manual_spore_manager 
                and getattr(self.manual_spore_manager, "ghost_tree_dt_vector", None) is not None
                and getattr(self.manual_spore_manager, "ghost_dt_baseline", None) is not None
                and self.dt_manager):
                
                import numpy as np
                old_vec = self.manual_spore_manager.ghost_tree_dt_vector
                old_base = float(self.manual_spore_manager.ghost_dt_baseline)
                new_base = float(self.dt_manager.get_dt())
                print(f"[InputManager._on_dt_changed] scale ghost dt-vector: baseline {old_base:.6f} -> {new_base:.6f}")

                if old_base > 0:
                    k = new_base / old_base
                    before_sample = old_vec.copy()
                    scaled = np.sign(old_vec) * (np.abs(old_vec) * k)
                    self.manual_spore_manager.ghost_tree_dt_vector = scaled
                    self.manual_spore_manager.ghost_dt_baseline = new_base

                    # Диагностика: покажем пару первых элементов ДО/ПОСЛЕ
                    def _fmt(arr):
                        return "[" + ", ".join(f"{x:+.6f}" for x in arr[:4]) + (" ..." if len(arr) > 4 else "") + "]"
                    print(f"[InputManager._on_dt_changed] k={k:.6f}")
                    print(f"[InputManager._on_dt_changed] dt_vec before: {_fmt(before_sample)}")
                    print(f"[InputManager._on_dt_changed] dt_vec after : {_fmt(scaled)}")
                else:
                    print("[InputManager._on_dt_changed] WARNING: ghost_dt_baseline == 0, skip scaling.")

            # 1) Пересобрать призрачные предсказания (дети/внуки двигаются)
            if self.manual_spore_manager and hasattr(self.manual_spore_manager, "_update_predictions"):
                self.manual_spore_manager.prediction_manager.debug_ghost_tree = True  # временно включим лог
                self.manual_spore_manager._update_predictions()
                self.manual_spore_manager.prediction_manager.debug_ghost_tree = False

            # 2) Применить новый лимит длины для ВСЕХ линков
            max_len = None
            if self.dt_manager and hasattr(self.dt_manager, "get_max_link_length"):
                max_len = self.dt_manager.get_max_link_length()

            # 2a) Обычные линки (через SporeManager)
            if self.spore_manager and hasattr(self.spore_manager, "update_links_max_length"):
                self.spore_manager.update_links_max_length(max_len)

            # 2b) Призрачные линки (через PredictionManager)
            if self.manual_spore_manager and hasattr(self.manual_spore_manager, "prediction_manager"):
                pm = self.manual_spore_manager.prediction_manager
                if hasattr(pm, "update_links_max_length"):
                    pm.update_links_max_length(max_len)
                # На всякий случай обновим геометрию каждого призрачного линка
                if hasattr(pm, "prediction_links"):
                    for link in pm.prediction_links:
                        try:
                            link.update_geometry()
                        except Exception:
                            pass

            # 3) Обновим общую трансформацию зума, чтобы всё подхватилось
            if self.zoom_manager and hasattr(self.zoom_manager, "update_transform"):
                self.zoom_manager.update_transform()

        except Exception as e:
            print(f"[InputManager] _on_dt_changed error: {e}")
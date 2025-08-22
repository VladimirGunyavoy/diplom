from ursina import held_keys, mouse
import time
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
        if held_keys['f']:
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
        
        if key == 'u':
            if self.scene_setup and hasattr(self.scene_setup, 'frame'):
                self.scene_setup.frame.toggle_visibility()
            return
            
        # 3. Управление масштабированием
        if self.zoom_manager:
                # Проверяем нажат ли Ctrl
            ctrl_pressed = held_keys['left control'] or held_keys['right control']
            
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
        if  held_keys['c'] and self.spore_manager and held_keys['left control']:
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
                # Принудительно обновляем предсказания
                if self.manual_spore_manager:
                    self.manual_spore_manager.preview_manager.force_update_predictions()
            elif key == '6':
                self.spore_manager.adjust_min_radius(1.2)    # Увеличить радиус (×1.2)
                # Принудительно обновляем предсказания
                if self.manual_spore_manager:
                    self.manual_spore_manager.preview_manager.force_update_predictions()
        
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
        if key == 'p':
            if self.manual_spore_manager:
                self.manual_spore_manager.update_ghost_tree_with_optimal_pairs()
            return
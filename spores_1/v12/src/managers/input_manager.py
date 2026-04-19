from ursina import held_keys
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
                 cost_visualizer: Optional['CostVisualizer'] = None):
        
        self.scene_setup: Optional[SceneSetup] = scene_setup
        self.zoom_manager: Optional[ZoomManager] = zoom_manager
        self.spore_manager: Optional[SporeManager] = spore_manager
        self.spawn_area_manager: Optional[SpawnAreaManager] = spawn_area_manager
        self.param_manager: Optional[ParamManager] = param_manager
        self.ui_setup: Optional[UI_setup] = ui_setup
        self.angel_manager: Optional['AngelManager'] = angel_manager
        self.cost_visualizer: Optional['CostVisualizer'] = cost_visualizer

        # Настройки для генерации спор по клавише 'f'
        self.f_key_down_time: float = 0
        self.long_press_threshold: float = 0.4
        self.spawn_interval: float = 0.1
        self.next_spawn_time: float = 0

    def update(self) -> None:
        """
        Этот метод должен вызываться каждый кадр для обработки непрерывного ввода.
        """
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
            # Сбрасываем таймер, если клавиша отпущена
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
            
        # 3. Управление масштабированием
        if self.zoom_manager:
            if key == 'e': 
                self.zoom_manager.zoom_in()
            elif key == 't': 
                self.zoom_manager.zoom_out()
            elif key == 'r': 
                self.zoom_manager.reset_zoom()
            elif key == '1': 
                self.zoom_manager.increase_spores_scale()
            elif key == '2': 
                self.zoom_manager.decrease_spores_scale()
            
        # 4. Управление параметром
        if self.param_manager:
            if key == 'z': 
                self.param_manager.increase()
            elif key == 'x': 
                self.param_manager.decrease()

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
        
        # 8. Переключение поверхности стоимости
        if key == 'c':
            if self.cost_visualizer:
                self.cost_visualizer.toggle()
            else:
                always_print("⚠️ Cost Visualizer не найден")

    # def handle_input(self, key):
    #     """
    #     Обрабатывает нажатия клавиш. 
    #     Этот метод предназначен для передачи в `input()` Ursina.
    #     """
    #     # --- Управление приложением ---
    #     if key == 'q':
    #         application.quit()

    #     # --- Управление симуляцией ---
    #     if key == '8':  # Клавиша для спавна новой споры
    #         self.engine.spawn_spore()

    #     # --- Управление видимостью ---
    #     if key == '5':  # Клавиша для поверхности стоимости
    #         self.is_cost_surface_visible = not self.is_cost_surface_visible
    #         self.visual_manager.toggle_cost_surface(self.is_cost_surface_visible)
        
    #     if key == '6':  # Клавиша для области спавна
    #         self.is_spawn_area_visible = not self.is_spawn_area_visible
    #         self.visual_manager.toggle_spawn_area(self.is_spawn_area_visible)
            
    #     if key == '7':  # Клавиша для "призраков"
    #         self.is_ghosts_visible = not self.is_ghosts_visible
    #         self.visual_manager.toggle_ghosts(self.is_ghosts_visible) 
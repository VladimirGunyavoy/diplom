from ursina import *
from .spawn_area_manager import SpawnAreaManager

class InputManager:
    """
    Централизованный класс для обработки всего пользовательского ввода.
    Принимает ссылки на все необходимые менеджеры и объекты сцены,
    и делегирует им команды в зависимости от нажатой клавиши.
    """
    def __init__(self, scene_setup=None, zoom_manager=None, spore_manager=None, 
                 spawn_area_manager=None, param_manager=None, ui_setup=None):
        self.scene_setup = scene_setup
        self.zoom_manager = zoom_manager
        self.spore_manager = spore_manager
        self.spawn_area_manager = spawn_area_manager
        self.param_manager = param_manager
        self.ui_setup = ui_setup

    def handle_input(self, key):
        """
        Основной метод-обработчик. Вызывается из главного цикла приложения.
        """
        # Отладка: выводим полученную клавишу и состояние курсора
        print(f"[InputManager] handle_input called with key: '{key}'. Cursor locked: {self.scene_setup.cursor_locked if self.scene_setup else 'N/A'}")

        # Базовое управление сценой (выход) - работает всегда
        if key == 'q' or key == 'escape':
            application.quit()
            return

        # Остальные команды работают только когда курсор свободен
        # if self.scene_setup and self.scene_setup.cursor_locked:
        #     # Отладка: выводим, почему мы прерываем выполнение
        #     print(f"[InputManager] Aborting: Cursor is locked.")
        #     return
        
        # Этот принт был в файле, но я его оставлю для дополнительной отладки
        print(f"[InputManager] Passed lock check. Processing key '{key}'...")
        # 1. UI команды (самый высокий приоритет)
        if self.ui_setup:
            # Отладка: что возвращает обработчик UI?
            print(f"[InputManager] Calling ui_setup.handle_demo_commands('{key}')...")
            processed = self.ui_setup.handle_demo_commands(key)
            print(f"[InputManager] ...ui_setup.handle_demo_commands returned: {processed}")
            # handle_demo_commands вернет True, если команда была обработана
            if processed:
                return

        # 2. Управление спорами
        if held_keys['f']:
            if self.spore_manager:
                self.spore_manager.generate_new_spore()
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
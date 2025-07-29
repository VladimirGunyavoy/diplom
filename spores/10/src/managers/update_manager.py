import time

class UpdateManager:
    """
    Централизованный класс для вызова методов обновления каждый кадр.
    Воспроизводит логику из стабильной версии проекта.
    """
    def __init__(self, scene_setup=None, zoom_manager=None, param_manager=None, ui_setup=None):
        self.scene_setup = scene_setup
        self.zoom_manager = zoom_manager
        self.param_manager = param_manager
        self.ui_setup = ui_setup
    
    def update_all(self):
        """
        Основной метод, который должен вызываться каждый кадр из главного цикла.
        """
        dt = time.dt

        if self.scene_setup:
            self.scene_setup.update(dt)
        
        if self.zoom_manager:
            self.zoom_manager.identify_invariant_point()
        
        if self.param_manager:
            self.param_manager.update()
            
        if self.ui_setup:
            self.ui_setup.update() 
from ursina import window, color
from typing import Optional

class WindowManager:
    """
    Простой класс для управления настройками окна Ursina.
    """
    
    # Настройки для разных мониторов
    MONITORS = {
        "main": {"size": (1920, 1080), "position": (0, 0)},
        "top": {"size": (1920, 1080), "position": (0, -1080)},
        "left": {"size": (1800, 950), "position": (-1850, 200)},
        "down": {"size": (3000, 1700), "position": (-500, 1500)}
    }
    
    def __init__(self, title: str = "Ursina App", monitor: str = "main", fullscreen: bool = False):
        """
        Инициализирует менеджер окна.
        
        Args:
            title (str): Заголовок окна.
            monitor (str): Тип монитора ("main", "top", "left", "down").
            fullscreen (bool): Запускать ли в полноэкранном режиме.
        """
        self.current_monitor = monitor  # Сохраняем текущий монитор
        self.fullscreen = fullscreen
        window.title = title
        
        # Применяем настройки монитора
        if fullscreen:
            # В полноэкранном режиме используем размер экрана
            window.fullscreen = True
        else:
            config = self.MONITORS.get(monitor, self.MONITORS["main"])
            window.size = config["size"]
            window.position = config["position"]

    
    def get_current_monitor(self) -> str:
        """Возвращает название текущего монитора."""
        return self.current_monitor
    
    def set_size(self, size: tuple) -> None:
        """Устанавливает размер окна."""
        window.size = size
    
    def set_position(self, position: tuple) -> None:
        """Устанавливает позицию окна."""
        window.position = position
    
    def set_background_color(self, a_color: color) -> None:
        """Устанавливает цвет фона."""
        window.color = a_color
    
    def toggle_fullscreen(self) -> None:
        """Переключает полноэкранный режим."""
        self.fullscreen = not self.fullscreen
        window.fullscreen = self.fullscreen
        
        if not self.fullscreen:
            # Возвращаем к настройкам монитора при выходе из полноэкранного режима
            config = self.MONITORS.get(self.current_monitor, self.MONITORS["main"])
            window.size = config["size"]
            window.position = config["position"]
    
    def set_fullscreen(self, fullscreen: bool) -> None:
        """Устанавливает полноэкранный режим."""
        self.fullscreen = fullscreen
        window.fullscreen = fullscreen
        
        if not fullscreen:
            # Возвращаем к настройкам монитора при выходе из полноэкранного режима
            config = self.MONITORS.get(self.current_monitor, self.MONITORS["main"])
            window.size = config["size"]
            window.position = config["position"]
    
    def is_fullscreen(self) -> bool:
        """Возвращает True, если окно в полноэкранном режиме."""
        return self.fullscreen

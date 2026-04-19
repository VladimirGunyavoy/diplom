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
        "right": {"size": (1920, 1080), "position": (1920, 0)},
        "down": {"size": (3000, 1700), "position": (-500, 1500)}
    }
    
    def __init__(self, title: str = "Ursina App", monitor: str = "main"):
        """
        Инициализирует менеджер окна.
        
        Args:
            title (str): Заголовок окна.
            monitor (str): Тип монитора ("main", "top", "right", "down").
        """
        self.current_monitor = monitor  # Сохраняем текущий монитор
        window.title = title
        
        # Применяем настройки монитора
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

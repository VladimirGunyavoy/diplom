from ursina import window, color
from typing import Optional

class WindowManager:
    """
    Класс для управления настройками окна Ursina.
    """
    def __init__(self, title: str = "Ursina App", size: Optional[tuple] = None, 
                monitor: str = "top"):
        """
        Инициализирует менеджер окна.
        
        Args:
            title (str): Заголовок окна.
            size (tuple, optional): Размер окна (ширина, высота).
        """
        window.title = title
        self.set_size(size)
        # self.monitor = 
        self.down_monitor_size = (3000, 1700)
        self.down_monitor_position = (-500, 1500)

        # self right_

        if down_monitor:
            window.size = self.down_monitor_size
            window.position = self.down_monitor_position

    def set_size(self, size: Optional[tuple]) -> None:
        """Устанавливает размер окна."""
        if size:
            window.size = size

    def set_position(self, position: Optional[tuple]) -> None:
        """Устанавливает позицию окна."""
        if position:
            window.position = position

    def set_background_color(self, a_color: color) -> None:
        """Устанавливает цвет фона."""
        window.color = a_color

from ursina import window, color
from typing import Optional

class WindowManager:
    """
    Класс для управления настройками окна Ursina.
    """
    def __init__(self, title: str = "Ursina App", size: Optional[tuple] = None):
        """
        Инициализирует менеджер окна.
        
        Args:
            title (str): Заголовок окна.
            size (tuple, optional): Размер окна (ширина, высота).
        """
        window.title = title
        self.set_size(size)

    def set_size(self, size: Optional[tuple]) -> None:
        """Устанавливает размер окна."""
        if size:
            window.size = size

    def set_background_color(self, a_color: color) -> None:
        """Устанавливает цвет фона."""
        window.color = a_color

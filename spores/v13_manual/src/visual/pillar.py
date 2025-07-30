from ..utils.scalable import Scalable
from typing import Any

class Pillar(Scalable):
    """
    Класс для столба, который может масштабироваться с помощью ZoomManager.
    Наследуется от Scalable, чтобы иметь метод apply_transform.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs) 
from src.utils.scalable import Scalable

class Pillar(Scalable):
    """
    Класс для столба, который может масштабироваться с помощью ZoomManager.
    Наследуется от Scalable, чтобы иметь метод apply_transform.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
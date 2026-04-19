from ..visual.spore_visual import SporeVisual
from typing import List

class VisualManager:
    """
    Класс, который управляет всеми визуальными элементами симуляции.
    Создает, обновляет и удаляет 3D-объекты.
    """
    def __init__(self):
        self.spores: List[SporeVisual] = []
        # Другие визуальные элементы могут быть добавлены сюда

    def create_spore_visual(self, **kwargs) -> SporeVisual:
        """
        Создает визуальное представление споры.
        
        Returns:
            Экземпляр SporeVisual.
        """
        spore_visual = SporeVisual(**kwargs)
        self.spores.append(spore_visual)
        return spore_visual

    def update_visuals(self) -> None:
        """
        Обновляет все визуальные элементы.
        На данный момент вызывается извне.
        """
        # Здесь может быть логика обновления, если она потребуется
        pass

    def clear_all(self) -> None:
        """
        Удаляет все визуальные элементы со сцены.
        """
        for spore in self.spores:
            spore.destroy()
        self.spores = [] 
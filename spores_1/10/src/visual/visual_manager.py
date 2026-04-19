from ursina import Entity, destroy
from src.visual.spore_visual import SporeVisual

class VisualManager(Entity):
    """
    Главный визуальный менеджер.
    Этот класс является контейнером для всех визуальных компонентов симуляции
    и отвечает за их синхронизацию с логическим движком.
    """

    def __init__(self, engine, cost_visualizer, spawn_visualizer, ghost_visualizer, **kwargs):
        """
        Инициализирует VisualManager.

        Args:
            engine: Экземпляр SimulationEngine.
            cost_visualizer: Визуализатор поверхности стоимости.
            spawn_visualizer: Визуализатор области спавна.
            ghost_visualizer: Визуализатор "призраков".
        """
        super().__init__(**kwargs)
        
        self.engine = engine
        self.cost_visualizer = cost_visualizer
        self.spawn_visualizer = spawn_visualizer
        self.ghost_visualizer = ghost_visualizer

        # Устанавливаем родительские связи
        self.cost_visualizer.parent = self
        self.spawn_visualizer.parent = self
        self.ghost_visualizer.parent = self

        self.visual_spores: dict[str, SporeVisual] = {}

    def toggle_cost_surface(self, is_enabled: bool):
        """Включает/выключает отображение поверхности стоимости."""
        self.cost_visualizer.enabled = is_enabled

    def toggle_spawn_area(self, is_enabled: bool):
        """Включает/выключает отображение области спавна."""
        self.spawn_visualizer.enabled = is_enabled
        
    def toggle_ghosts(self, is_enabled: bool):
        """Включает/выключает отображение призраков."""
        self.ghost_visualizer.enabled = is_enabled

    def sync_with_engine(self):
        """
        Синхронизирует визуальное представление с состоянием SimulationEngine.
        Создает, удаляет и обновляет визуальные споры.
        """
        engine_spores = self.engine.get_all_spore_states()
        engine_spore_ids = set(engine_spores.keys())
        visual_spore_ids = set(self.visual_spores.keys())

        # 1. Удалить визуальные споры, которых больше нет в движке
        spores_to_remove = visual_spore_ids - engine_spore_ids
        for spore_id in spores_to_remove:
            destroy(self.visual_spores[spore_id])
            del self.visual_spores[spore_id]

        # 2. Создать новые визуальные споры
        spores_to_add = engine_spore_ids - visual_spore_ids
        for spore_id in spores_to_add:
            state = engine_spores[spore_id]
            pos_2d = state['position']
            self.visual_spores[spore_id] = SporeVisual(
                position=(pos_2d[0], 0, pos_2d[1]), 
                parent=self
            )

        # 3. Обновить существующие визуальные споры
        for spore_id in self.visual_spores:
            state = engine_spores[spore_id]
            pos_2d = state['position']
            self.visual_spores[spore_id].position = (pos_2d[0], 0, pos_2d[1]) 
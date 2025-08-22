from ursina import *
from typing import Optional
import time

# Предварительное объявление классов для аннотаций типов
from ..visual.scene_setup import SceneSetup
from ..managers.zoom_manager import ZoomManager
from ..managers.param_manager import ParamManager
from ..visual.ui_setup import UI_setup
from ..managers.input_manager import InputManager

# Forward declaration для ManualSporeManager (избегаем циклических импортов)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..managers.manual_spore_manager import ManualSporeManager

class UpdateManager:
    """
    Централизованный класс для вызова методов обновления каждый кадр.
    Воспроизводит логику из стабильной версии проекта.
    """
    def __init__(self, 
                 scene_setup: Optional[SceneSetup] = None, 
                 zoom_manager: Optional[ZoomManager] = None, 
                 param_manager: Optional[ParamManager] = None, 
                 ui_setup: Optional[UI_setup] = None, 
                 input_manager: Optional[InputManager] = None,
                 manual_spore_manager: Optional["ManualSporeManager"] = None):
        
        self.scene_setup: Optional[SceneSetup] = scene_setup
        self.zoom_manager: Optional[ZoomManager] = zoom_manager
        self.param_manager: Optional[ParamManager] = param_manager
        self.ui_setup: Optional[UI_setup] = ui_setup
        self.input_manager: Optional[InputManager] = input_manager
        self.manual_spore_manager: Optional["ManualSporeManager"] = manual_spore_manager
    
    def update_all(self) -> None:
        """
        Основной метод, который должен вызываться каждый кадр из главного цикла.
        """
        # Проверяем manual_spore_manager перед обновлением
        if self.manual_spore_manager:
            try:
                # Проверяем есть ли проблемные preview объекты
                preview_spore = self.manual_spore_manager.preview_manager.get_preview_spore()
                if preview_spore and (not hasattr(preview_spore, 'parent') or preview_spore.parent is None):
                    print(f"⚠️ UpdateManager: Preview spore has no parent, removing...")
                    self.manual_spore_manager.preview_manager.clear_all()

                # Проверяем prediction objects
                pred_viz = self.manual_spore_manager.preview_manager.prediction_visualizers
                pred_links = self.manual_spore_manager.preview_manager.prediction_links

                # Проверяем целостность объектов
                bad_objects = []
                for viz in pred_viz:
                    if hasattr(viz, 'ghost_spore') and viz.ghost_spore:
                        if not hasattr(viz.ghost_spore, 'parent') or viz.ghost_spore.parent is None:
                            bad_objects.append(viz)

                for link in pred_links:
                    if not hasattr(link, 'parent') or link.parent is None:
                        bad_objects.append(link)

                if bad_objects:
                    print(f"⚠️ UpdateManager: Найдены плохие prediction объекты: {len(bad_objects)}")
                    self.manual_spore_manager.preview_manager.clear_all()

            except Exception as e:
                print(f"❌ Ошибка в проверке UpdateManager: {e}")

        if self.input_manager:
            self.input_manager.update()

        if self.scene_setup:
            self.scene_setup.update(time.dt)

        if self.zoom_manager:
            self.zoom_manager.identify_invariant_point()  # Для зума и UI

        # v13_manual: обновляем позицию курсора для превью споры
        if self.manual_spore_manager:
            self.manual_spore_manager.update_cursor_position()  # Без параметров - сам вычислит

        if self.param_manager:
            self.param_manager.update()

        if self.ui_setup:
            self.ui_setup.update() 
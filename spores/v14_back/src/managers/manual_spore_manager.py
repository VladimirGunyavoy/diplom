from typing import Optional, Tuple, List
import numpy as np
from ursina import Vec3, color, destroy, mouse, camera, raycast, Vec2

from ..core.spore import Spore
from ..managers.zoom_manager import ZoomManager
from ..managers.spore_manager import SporeManager
from ..managers.color_manager import ColorManager
from ..logic.pendulum import PendulumSystem
from ..visual.prediction_visualizer import PredictionVisualizer
from ..visual.link import Link


class ManualSporeManager:
    """
    Менеджер для ручного создания спор с превью по курсору.
    
    Функциональность:
    - Показывает полупрозрачную спору в позиции курсора мыши
    - Отображает предсказания при min/max управлении
    - Создает споры + цепочки родителей по ЛКМ
    """
    
    def __init__(self, 
                 spore_manager: SporeManager,
                 zoom_manager: ZoomManager,
                 pendulum: PendulumSystem,
                 color_manager: ColorManager,
                 config: dict):
        
        self.spore_manager = spore_manager
        self.zoom_manager = zoom_manager
        self.pendulum = pendulum
        self.color_manager = color_manager
        self.config = config
        
        # Настройки превью
        self.preview_enabled = True
        self.preview_alpha = 0.5  # Полупрозрачность
        
        # Превью спора
        self.preview_spore: Optional[Spore] = None
        self.preview_position_2d = np.array([0.0, 0.0], dtype=float)
        
        # Предсказания min/max управления
        self.prediction_visualizers: List[PredictionVisualizer] = []
        self.prediction_links: List[Link] = []  # Линки от превью споры к призракам
        self.created_links: List[Link] = []
        self.show_predictions = True
        
        # Получаем границы управления из маятника
        control_bounds = self.pendulum.get_control_bounds()
        self.min_control = float(control_bounds[0])
        self.max_control = float(control_bounds[1])

        self._link_counter = 0
        self._spore_counter = 0
        
        print(f"   ✓ Manual Spore Manager создан (управление: {self.min_control} .. {self.max_control})")

    def _get_next_link_id(self) -> int:
        """Возвращает уникальный ID для линка"""
        self._link_counter += 1
        return self._link_counter
    
    def _get_next_spore_id(self) -> int:
        """Возвращает уникальный ID для споры"""
        self._spore_counter += 1
        return self._spore_counter
    
    def get_mouse_world_position(self) -> Optional[Tuple[float, float]]:
        """
        Получает позицию курсора мыши с правильной трансформацией зума.
        
        Алгоритм:
        1. Определить точку на которую смотрит камера (без зума)
        2. Отнять позицию frame origin_cube
        3. Поделить на common_scale
        
        Returns:
            (x, z) координаты курсора в правильной системе координат
        """
        try:
            # 1. Получаем точку взгляда камеры (без зума)
            look_point_x, look_point_z = self.zoom_manager.identify_invariant_point()
            
            # 2. Получаем трансформированную позицию origin_cube из frame
            frame = getattr(self.zoom_manager.scene_setup, 'frame', None)
            if frame and hasattr(frame, 'origin_cube'):
                # Используем обычную position (после трансформаций) а не real_position
                origin_pos = frame.origin_cube.position
                if hasattr(origin_pos, 'x'):
                    origin_x, origin_z = origin_pos.x, origin_pos.z
                else:
                    origin_x, origin_z = origin_pos[0], origin_pos[2]
            else:
                # Fallback если frame не найден
                origin_x, origin_z = 0.0, 0.0
            
            # 3. Получаем масштаб трансформации
            transform_scale = getattr(self.zoom_manager, 'a_transformation', 1.0)
            
            # 4. ПРАВИЛЬНАЯ ФОРМУЛА: (look_point - frame_origin_cube) / scale
            corrected_x = (look_point_x - origin_x) / transform_scale
            corrected_z = (look_point_z - origin_z) / transform_scale
            
            return (corrected_x, corrected_z)
                
        except Exception as e:
            print(f"Ошибка определения позиции мыши: {e}")
            return (0.0, 0.0)
    
    def update_cursor_position(self) -> None:
        """
        Обновляет позицию превью споры по позиции курсора мыши.
        """
        if not self.preview_enabled:
            return
        
        # Получаем правильную позицию курсора мыши
        mouse_pos = self.get_mouse_world_position()
        if mouse_pos is None:
            return
            
        # Обновляем позицию
        self.preview_position_2d[0] = mouse_pos[0]
        self.preview_position_2d[1] = mouse_pos[1]
        
        # Создаем или обновляем превью спору
        self._update_preview_spore()
        
        # Обновляем предсказания
        if self.show_predictions:
            self._update_predictions()
    
    def set_preview_enabled(self, enabled: bool) -> None:
        """Включает/выключает превью спор."""
        self.preview_enabled = enabled
        if not enabled:
            self._destroy_preview()
    
    def _update_preview_spore(self) -> None:
        """Создает или обновляет превью спору."""
        if not self.preview_spore:
            self._create_preview_spore()
        else:
            # Обновляем позицию существующей споры
            self.preview_spore.real_position = Vec3(
                self.preview_position_2d[0], 
                0.0, 
                self.preview_position_2d[1]
            )
            # Применяем трансформации zoom manager
            self.preview_spore.apply_transform(
                self.zoom_manager.a_transformation,
                self.zoom_manager.b_translation,
                spores_scale=self.zoom_manager.spores_scale
            )
    
    def _create_preview_spore(self) -> None:
        """Создает новую превью спору."""
        try:
            goal_position = self.config.get('spore', {}).get('goal_position', [0, 0])
            spore_config = self.config.get('spore', {})
            pendulum_config = self.config.get('pendulum', {})
            
            self.preview_spore = Spore(
                pendulum=self.pendulum,
                dt=pendulum_config.get('dt', 0.1),
                goal_position=goal_position,
                scale=spore_config.get('scale', 0.1),
                position=(self.preview_position_2d[0], 0.0, self.preview_position_2d[1]),
                color_manager=self.color_manager,
                is_ghost=True  # Делаем спору-призрак
            )
            
            base_color = self.color_manager.get_color('spore', 'default')
            
            # Способ 1: Используем атрибуты r, g, b
            try:
                self.preview_spore.color = (base_color.r, base_color.g, base_color.b, self.preview_alpha)
            except AttributeError:
                # Способ 2: Если это Vec4 или tuple
                try:
                    self.preview_spore.color = (base_color[0], base_color[1], base_color[2], self.preview_alpha)
                except (TypeError, IndexError):
                    # Способ 3: Fallback на стандартный цвет
                    self.preview_spore.color = (0.6, 0.4, 0.9, self.preview_alpha)  # Фиолетовый по умолчанию
                    print(f"   ⚠️ Использован fallback цвет для preview spore")
            
            # Регистрируем в zoom manager
            self.zoom_manager.register_object(self.preview_spore, name='manual_preview')
            
        except Exception as e:
            print(f"Ошибка создания превью споры: {e}")
    

    def _update_predictions(self) -> None:
        """Обновляет предсказания: 2 вперед (min/max) + 2 назад (min/max)."""
        if not self.preview_spore:
            return
            
        try:
            # Очищаем старые предсказания
            self._clear_predictions()
            
            dt = self.config.get('pendulum', {}).get('dt', 0.1)
            
            # 4 предсказания: 2 вперед + 2 назад
            prediction_configs = [
                # Вперед
                {'control': self.min_control, 'name': 'forward_min', 'color': 'ghost_min', 'direction': 'forward'},
                {'control': self.max_control, 'name': 'forward_max', 'color': 'ghost_max', 'direction': 'forward'},
                # Назад  
                {'control': self.min_control, 'name': 'backward_min', 'color': 'ghost_min', 'direction': 'backward'},
                {'control': self.max_control, 'name': 'backward_max', 'color': 'ghost_max', 'direction': 'backward'}
            ]
            
            for i, config in enumerate(prediction_configs):
                # Вычисляем позицию в зависимости от направления
                if config['direction'] == 'forward':
                    # Обычный шаг вперед
                    predicted_pos_2d = self.pendulum.scipy_rk45_step(
                        self.preview_position_2d, 
                        config['control'], 
                        dt
                    )
                else:  # backward
                    # Шаг назад во времени
                    predicted_pos_2d = self.pendulum.scipy_rk45_step_backward(
                        self.preview_position_2d, 
                        config['control'], 
                        dt
                    )
                
                # Создаем визуализатор предсказания
                prediction_viz = PredictionVisualizer(
                    parent_spore=self.preview_spore,
                    color_manager=self.color_manager,
                    zoom_manager=self.zoom_manager,
                    cost_function=None,  # Не показываем cost для предсказаний
                    config={
                        'spore': {'show_ghosts': True}, 
                        'angel': {'show_angels': False, 'show_pillars': False}
                    },
                    spore_id=f'manual_prediction_{config["name"]}'
                )
                
                # Устанавливаем цвет призрака
                if prediction_viz.ghost_spore:
                    base_color = self.color_manager.get_color('spore', config['color'])
                    
                    # Для призраков назад делаем цвет более тусклым
                    if config['direction'] == 'backward':
                        # Уменьшаем яркость для призраков назад
                        if hasattr(base_color, 'r'):
                            r, g, b = base_color.r * 0.6, base_color.g * 0.6, base_color.b * 0.6
                            prediction_viz.ghost_spore.color = (r, g, b, 0.7)
                        else:
                            # Fallback
                            prediction_viz.ghost_spore.color = (0.3, 0.3, 0.6, 0.7)  # Тусклый синий
                    else:
                        # Обычный цвет для призраков вперед
                        prediction_viz.ghost_spore.color = base_color
                
                # Обновляем позицию предсказания
                prediction_viz.update(predicted_pos_2d)
                
                # # Создаем линк от превью споры к призраку
                # if prediction_viz.ghost_spore:
                #     prediction_link = Link(
                #         parent_spore=self.preview_spore,
                #         child_spore=prediction_viz.ghost_spore,
                #         color_manager=self.color_manager,
                #         zoom_manager=self.zoom_manager,
                #         config=self.config
                #     )

                # Создаем линк от превью споры к призраку
                if prediction_viz.ghost_spore:
                    # Для обратного направления меняем местами parent и child
                    if config['direction'] == 'forward':
                        # Вперед: превью → будущее состояние  
                        parent_spore = self.preview_spore
                        child_spore = prediction_viz.ghost_spore
                    else:  # backward
                        # Назад: прошлое состояние → превью (показываем откуда пришли)
                        parent_spore = prediction_viz.ghost_spore  
                        child_spore = self.preview_spore
                    
                    prediction_link = Link(
                        parent_spore=parent_spore,
                        child_spore=child_spore,
                        color_manager=self.color_manager,
                        zoom_manager=self.zoom_manager,
                        config=self.config
                    )
                    
                    # Устанавливаем цвет линка в зависимости от управления  
                    if config['direction'] == 'forward':
                        # Обычные цвета для линков вперед
                        if 'min' in config['name']:
                            link_color_name = 'ghost_min'  # Синий для min
                        else:  # max
                            link_color_name = 'ghost_max'  # Красный для max
                    else:  # backward
                        # Те же цвета для линков назад
                        if 'min' in config['name']:
                            link_color_name = 'ghost_min'  # Синий для min
                        else:  # max
                            link_color_name = 'ghost_max'  # Красный для max
                        
                    prediction_link.color = self.color_manager.get_color('link', link_color_name)
                    
                    # Для линков назад делаем более тонкими или пунктирными
                    if config['direction'] == 'backward':
                        # Можно добавить дополнительную стилизацию линков назад
                        pass
                    
                    # Обновляем геометрию и регистрируем в zoom manager
                    prediction_link.update_geometry()
                    self.zoom_manager.register_object(
                        prediction_link, 
                        f'manual_prediction_link_{config["name"]}'
                    )
                    
                    self.prediction_links.append(prediction_link)
                
                self.prediction_visualizers.append(prediction_viz)
                
        except Exception as e:
            print(f"Ошибка обновления предсказаний: {e}")
            import traceback
            traceback.print_exc()

    def _clear_predictions(self) -> None:
        """Очищает все предсказания и их линки."""
        # Очищаем визуализаторы предсказаний
        for viz in self.prediction_visualizers:
            viz.destroy()
        self.prediction_visualizers.clear()
        
        # Очищаем линки предсказаний
        for i, link in enumerate(self.prediction_links):
            link_name = f'manual_prediction_link_{["min", "max"][i] if i < 2 else i}'
            try:
                self.zoom_manager.unregister_object(link_name)
            except:
                pass  # Игнорируем ошибки если объект уже не зарегистрирован
            destroy(link)
        self.prediction_links.clear()
    
    def create_spore_at_cursor(self) -> Optional[List[Spore]]:
        """
        Создает полную семью спор:
        - Центральная спора в позиции курсора
        - 2 дочерние споры (forward min/max control)  
        - 2 родительские споры (backward min/max control)
        - Все соединительные линки с правильными цветами
        
        Returns:
            Список созданных спор [center, forward_min, forward_max, backward_min, backward_max] или None при ошибке
        """
        if not self.preview_enabled or not self.preview_spore:
            return None
            
        try:
            goal_position = self.config.get('spore', {}).get('goal_position', [0, 0])
            spore_config = self.config.get('spore', {})
            pendulum_config = self.config.get('pendulum', {})
            dt = pendulum_config.get('dt', 0.1)
            
            created_spores = []
            created_links = []
            
            # 1. Создаем ЦЕНТРАЛЬНУЮ спору в позиции курсора
            center_id = self._get_next_spore_id()
            center_spore = Spore(
                pendulum=self.pendulum,
                dt=dt,
                goal_position=goal_position,
                scale=spore_config.get('scale', 0.1),
                position=(self.preview_position_2d[0], 0.0, self.preview_position_2d[1]),
                color_manager=self.color_manager,
                config=spore_config
            )
            
            # Добавляем центральную спору в систему
            self.spore_manager.add_spore_manual(center_spore)
            self.zoom_manager.register_object(center_spore, f'manual_center_{center_id}')
            created_spores.append(center_spore)
            print(f"   ✓ Создана центральная спора в позиции ({self.preview_position_2d[0]:.3f}, {self.preview_position_2d[1]:.3f})")
            
            # 2. Создаем ДОЧЕРНИЕ споры (forward) + РОДИТЕЛЬСКИЕ споры (backward)
            spore_configs = [
                # Дочерние (forward)
                {'control': self.min_control, 'name': 'forward_min', 'color': 'ghost_min', 'direction': 'forward'},
                {'control': self.max_control, 'name': 'forward_max', 'color': 'ghost_max', 'direction': 'forward'},
                # Родительские (backward) 
                {'control': self.min_control, 'name': 'backward_min', 'color': 'ghost_min', 'direction': 'backward'},
                {'control': self.max_control, 'name': 'backward_max', 'color': 'ghost_max', 'direction': 'backward'}
            ]
            
            for config in spore_configs:
                child_id = self._get_next_spore_id()
                
                # Вычисляем позицию в зависимости от направления
                if config['direction'] == 'forward':
                    # Обычный шаг вперед
                    child_pos_2d = self.pendulum.scipy_rk45_step(
                        self.preview_position_2d, 
                        config['control'], 
                        dt
                    )
                else:  # backward
                    # Шаг назад во времени
                    child_pos_2d = self.pendulum.scipy_rk45_step_backward(
                        self.preview_position_2d, 
                        config['control'], 
                        dt
                    )
                
                # Создаем спору
                child_spore = Spore(
                    pendulum=self.pendulum,
                    dt=dt,
                    goal_position=goal_position,
                    scale=spore_config.get('scale', 0.1),
                    position=(child_pos_2d[0], 0.0, child_pos_2d[1]),
                    color_manager=self.color_manager,
                    config=spore_config
                )
                
                # Добавляем спору в систему БЕЗ автоматических призраков
                self.spore_manager.add_spore_manual(child_spore)
                self.zoom_manager.register_object(child_spore, f'manual_{config["name"]}_{child_id}')
                
                # Переопределяем управление на конкретное значение
                child_spore.logic.optimal_control = np.array([config['control']])
                
                created_spores.append(child_spore)
                print(f"   ✓ Создана спора {config['name']} в позиции ({child_pos_2d[0]:.3f}, {child_pos_2d[1]:.3f}) с управлением {config['control']:.2f}")
                
                # 3. Создаем ЛИНК с правильным направлением и цветом
                link_id = self._get_next_link_id()
                
                # Определяем направление стрелки
                if config['direction'] == 'forward':
                    # Вперед: центр → дочерняя
                    parent_spore = center_spore
                    child_link_spore = child_spore
                else:  # backward
                    # Назад: родительская → центр (показываем откуда пришли)
                    parent_spore = child_spore
                    child_link_spore = center_spore
                
                # Создаем линк
                spore_link = Link(
                    parent_spore=parent_spore,
                    child_spore=child_link_spore,
                    color_manager=self.color_manager,
                    zoom_manager=self.zoom_manager,
                    config=self.config
                )
                
                # Устанавливаем цвет линка в зависимости от управления
                if 'min' in config['name']:
                    # Синий цвет для минимального управления
                    link_color_name = 'ghost_min'
                else:  # max
                    # Красный цвет для максимального управления  
                    link_color_name = 'ghost_max'
                
                spore_link.color = self.color_manager.get_color('link', link_color_name)
                
                # Обновляем геометрию и регистрируем линк
                spore_link.update_geometry()
                self.zoom_manager.register_object(spore_link, f'manual_link_{config["name"]}_{link_id}')
                
                created_links.append(spore_link)
                self.created_links.append(spore_link)
                
                print(f"   ✓ Создан {link_color_name} линк для {config['name']} (направление: {config['direction']})")
            
            print(f"   🎯 Создано ВСЕГО: {len(created_spores)} спор + {len(created_links)} линков")
            print(f"   📊 Состав: 1 центральная + 2 дочерние (forward) + 2 родительские (backward)")
            
            return created_spores
            
        except Exception as e:
            print(f"Ошибка создания семьи спор: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _destroy_preview(self) -> None:
        """Уничтожает превью спору, предсказания и их линки."""
        if self.preview_spore:
            self.zoom_manager.unregister_object('manual_preview')
            destroy(self.preview_spore)
            self.preview_spore = None
            
        self._clear_predictions()  # Это очистит и визуализаторы, и линки
    
    def destroy(self) -> None:
        """Очищает все ресурсы менеджера."""
        self._destroy_preview()
        print("   ✓ Manual Spore Manager уничтожен") 

    def clear_all(self) -> None:
        """Очищает все ресурсы созданные ManualSporeManager."""
        print("🧹 ManualSporeManager: очистка всех ресурсов...")
        
        # 🆕 ДИАГНОСТИКА: Проверяем состояние до удаления
        print(f"   📊 Линков для удаления: {len(self.created_links)}")
        
        # 1. Уничтожаем все созданные линки
        for i, link in enumerate(self.created_links):
            try:
                # 🆕 ДИАГНОСТИКА: Проверяем состояние линка
                print(f"   🔍 Линк {i+1}: enabled={getattr(link, 'enabled', 'N/A')}, visible={getattr(link, 'visible', 'N/A')}")
                print(f"            parent={getattr(link, 'parent', 'N/A')}, model={getattr(link, 'model', 'N/A')}")
                
                # Пробуем разные способы удаления
                link.enabled = False  # Отключаем
                link.visible = False  # Скрываем
                link.parent = None    # Отвязываем от родителя
                
                destroy(link)  # Уничтожаем
                print(f"   ✅ Линк {i+1} обработан")
                
            except Exception as e:
                print(f"   ❌ Ошибка с линком {i+1}: {e}")
        
        # 🆕 ДИАГНОСТИКА: Проверяем глобальное состояние Ursina
        try:
            from ursina import scene, camera
            all_entities = [e for e in scene.entities if hasattr(e, 'model') and e.model]
            arrow_entities = [e for e in all_entities if 'arrow' in str(e.model).lower()]
            print(f"   📊 Всего entities в сцене: {len(scene.entities)}")
            print(f"   🏹 Entities со стрелками: {len(arrow_entities)}")
        except Exception as e:
            print(f"   ⚠️ Ошибка диагностики сцены: {e}")
        
        self.created_links.clear()

    def destroy(self) -> None:
        """Очищает все ресурсы менеджера."""
        self.clear_all()  # Используем новый метод
        print("   ✓ Manual Spore Manager уничтожен")
from ursina import destroy
import numpy as np
from typing import List, Optional, Dict, TYPE_CHECKING

from ..core.spore import Spore
from ..logic.pendulum import PendulumSystem
from ..visual.link import Link
from ..managers.color_manager import ColorManager
from ..visual.prediction_visualizer import PredictionVisualizer
from .zoom_manager import ZoomManager
from ..logic.optimizer import SporeOptimizer
from .param_manager import ParamManager
from .angel_manager import AngelManager
from ..utils.debug_output import DebugOutput, always_print, debug_print, evolution_print, candidate_print, trajectory_print

if TYPE_CHECKING:
    from ..logic.spawn_area import SpawnArea


class SporeManager:
    def __init__(self, pendulum: PendulumSystem, 
                 zoom_manager: ZoomManager, 
                 settings_param: Optional[ParamManager], 
                 color_manager: ColorManager, 
                 angel_manager: Optional[AngelManager] = None, 
                 config: Optional[Dict] = None,
                 spawn_area: Optional['SpawnArea'] = None):
        
        self.pendulum: PendulumSystem = pendulum
        self.zoom_manager: ZoomManager = zoom_manager
        self.settings_param: Optional[ParamManager] = settings_param
        self.color_manager: ColorManager = color_manager
        self.angel_manager: Optional[AngelManager] = angel_manager
        self.config: Dict = config if config is not None else {}
        self.spawn_area: Optional['SpawnArea'] = spawn_area
        
        # Инициализируем отладочный вывод
        self.debug = DebugOutput(self.config)
        
        self.optimizer: SporeOptimizer = SporeOptimizer(pendulum, config)

        self.objects: List[Spore] = []
        self.trajectories: Dict = {}
        self.spore_count: int = 0
        self.prediction_visualizers: List[PredictionVisualizer] = []
        self.ghost_link: Optional[Link] = None
        self.optimal_ghost_spore: Optional[Spore] = None  # Ссылка на оптимальную призрачную спору
        self.links: List[Link] = []
        self._next_spore_id: int = 0
        
        # Система кандидатских спор
        self.candidate_spores: List[Spore] = []  # Споры-кандидаты (белые прозрачные)
        self.min_radius: float = 0.3  # Минимальный радиус для дисков Пуассона
        self.candidate_count: int = 0

    def clear(self) -> None:
        """Удаляет все споры, связи и другие объекты со сцены."""
        for spore in self.objects:
            destroy(spore)
        for link in self.links:
            destroy(link)
        if self.ghost_link:
            destroy(self.ghost_link)
        for visualizer in self.prediction_visualizers:
            visualizer.destroy()

        self.objects = []
        self.links = []
        self.prediction_visualizers = []
        self.ghost_link = None
        self.optimal_ghost_spore = None
        self._next_spore_id = 0
        
        # Очищаем кандидатские споры
        for candidate in self.candidate_spores:
            # Удаляем из zoom_manager перед удалением объекта
            if hasattr(candidate, 'id'):
                self.zoom_manager.unregister_object(candidate.id)
            destroy(candidate)
        self.candidate_spores = []
        self.candidate_count = 0
        
        if self.angel_manager:
            self.angel_manager.clear_all()

    def clear_all_manual(self) -> None:
        """Полная очистка для v13_manual - удаляет все объекты КРОМЕ целевой споры."""
        
        print("🧹 ПОЛНАЯ ОЧИСТКА v13_manual...")
        
        # Очищаем все registered объекты из ZoomManager связанные со спорами
        if self.zoom_manager:
            # Получаем все ключи для удаления
            keys_to_remove = []
            registered_objects = getattr(self.zoom_manager, 'objects', {}) 
            
            for key in registered_objects:
                name_list = ['spore', 'link', 'ghost', 'predict', 'manual', 'angel', 'pillar']
                if any(pattern in key.lower() for pattern in name_list):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                try:
                    self.zoom_manager.unregister_object(key)
                    print(f"   ✓ Удален из ZoomManager: {key}")
                except Exception as e:
                    print(f"   ⚠️ Ошибка удаления {key}: {e}")
        
        # Стандартная очистка
        spores_to_remove = []
        for spore in self.objects:
            if not (hasattr(spore, 'is_goal') and spore.is_goal):
                spores_to_remove.append(spore)
        
        for spore in spores_to_remove:
            self.objects.remove(spore)
            destroy(spore)
            print(f"   ✓ Удалена обычная спора: {getattr(spore, 'id', 'unknown')}")
        
        # 2. Удаляем все связи
        for link in self.links:
            destroy(link)
        self.links = []
        print(f"   ✓ Удалено связей: {len(self.links)}")
        
        # 3. Удаляем ghost_link
        if self.ghost_link:
            destroy(self.ghost_link)
            self.ghost_link = None
            print("   ✓ Удалена призрачная связь")
        
        # 4. Удаляем prediction visualizers
        for visualizer in self.prediction_visualizers:
            visualizer.destroy()
        self.prediction_visualizers = []
        print(f"   ✓ Удалено visualizers: {len(self.prediction_visualizers)}")
        
        # 5. Сбрасываем optimal_ghost_spore
        self.optimal_ghost_spore = None

        # клод - лентяй!
        
        # 6. Очищаем кандидатские споры
        for candidate in self.candidate_spores:
            # Удаляем из zoom_manager перед удалением объекта
            if hasattr(candidate, 'id'):
                try:
                    self.zoom_manager.unregister_object(candidate.id)
                except:
                    pass
            destroy(candidate)
        self.candidate_spores = []
        self.candidate_count = 0
        print(f"   ✓ Удалено кандидатов: {len(self.candidate_spores)}")
        
        # 7. Очищаем angel_manager (если есть)
        if self.angel_manager:
            self.angel_manager.clear_ghosts()
            print("   ✓ Очищены ангелы")
        
        # 8. Подсчитываем что осталось
        remaining_spores = [s for s in self.objects if hasattr(s, 'is_goal') and s.is_goal]
        print(f"   💚 Сохранено целевых спор: {len(remaining_spores)}")
        
        print("🧹 Полная очистка завершена (целевые споры сохранены)")

    def add_spore(self, spore: Spore) -> None:
        """Добавляет спору в список управления."""
        if not isinstance(spore.id, int): # Присваиваем ID, если его еще нет
            spore.id = self._next_spore_id
            self._next_spore_id += 1

        optimal_control, optimal_dt = self.optimizer.find_optimal_step(spore)['x']
        optimal_control = np.array([optimal_control])
        spore.logic.optimal_control = optimal_control
        spore.logic.optimal_dt = optimal_dt
        
        # Расширенный отладочный вывод
        spore_type = ""
        if hasattr(spore, 'is_goal') and spore.is_goal:
            spore_type = "🎯 ЦЕЛЬ"
        elif hasattr(spore, 'is_candidate') and spore.is_candidate:
            spore_type = "⚪ КАНДИДАТ"
        elif hasattr(spore, 'is_ghost') and spore.is_ghost:
            spore_type = "👻 ПРИЗРАК"
        else:
            spore_type = "🔸 ОБЫЧНАЯ"
            
        debug_print(f"➕ {spore_type} спора {spore.id} добавлена:")
        debug_print(f"   📍 Позиция: {spore.calc_2d_pos()}")
        debug_print(f"   💰 Cost: {spore.logic.cost:.6f}")
        debug_print(f"   🎮 Управление: {optimal_control}")
        debug_print(f"   ⏱️  dt: {optimal_dt:.6f}")
        
        # Проверяем смерть споры (если optimal_dt = 0), но не для целевых спор
        if not (hasattr(spore, 'is_goal') and spore.is_goal):
            # spore.check_death()
            if not spore.is_alive():
                debug_print(f"🪦 Спора {spore.id} объявлена мертвой (dt = {optimal_dt}) - цвет изменен на серый")
        
        self.objects.append(spore)

        if self.angel_manager:
            self.angel_manager.on_spore_created(spore)

        self.sample_ghost_spores() # Обновляем призраков с фиксированными управлениями
        self.update_ghost_link()      # Обновляем призрачную связь

    def add_spore_manual(self, spore: Spore) -> None:
        """Добавляет спору БЕЗ автоматических призраков и обновлений (для v13_manual)."""
        if not isinstance(spore.id, int): # Присваиваем ID, если его еще нет
            spore.id = self._next_spore_id
            self._next_spore_id += 1

        optimal_control, optimal_dt = self.optimizer.find_optimal_step(spore)['x']
        optimal_control = np.array([optimal_control])
        spore.logic.optimal_control = optimal_control
        spore.logic.optimal_dt = optimal_dt
        
        # Расширенный отладочный вывод
        spore_type = ""
        if hasattr(spore, 'is_goal') and spore.is_goal:
            spore_type = "🎯 ЦЕЛЬ"
        elif hasattr(spore, 'is_candidate') and spore.is_candidate:
            spore_type = "⚪ КАНДИДАТ"
        elif hasattr(spore, 'is_ghost') and spore.is_ghost:
            spore_type = "👻 ПРИЗРАК"
        else:
            spore_type = "🔸 ОБЫЧНАЯ"
            
        debug_print(f"➕ {spore_type} спора {spore.id} добавлена (v13_manual):")
        debug_print(f"   📍 Позиция: {spore.calc_2d_pos()}")
        debug_print(f"   💰 Cost: {spore.logic.cost:.6f}")
        debug_print(f"   🎮 Управление: {optimal_control}")
        debug_print(f"   ⏱️  dt: {optimal_dt:.6f}")
        
        # Проверяем смерть споры (если optimal_dt = 0), но не для целевых спор
        if not (hasattr(spore, 'is_goal') and spore.is_goal):
            # spore.check_death()
            if not spore.is_alive():
                debug_print(f"🪦 Спора {spore.id} объявлена мертвой (dt = {optimal_dt}) - цвет изменен на серый")
        
        self.objects.append(spore)
        
        # НЕ вызываем:
        # - self.angel_manager.on_spore_created(spore) 
        # - self.sample_ghost_spores()
        # - self.update_ghost_link()

    def generate_new_spore(self) -> Optional[Spore]:
        """Создает новую спору на основе последней с проверкой пересечения траекторий."""
        if not self.objects:
            return None

        parent_spore = self.objects[-1]
        
        evolution_print(f"\n🚀 НАЧАЛО ГЕНЕРАЦИИ НОВОЙ СПОРЫ:")
        evolution_print(f"   👨‍👩‍👧‍👦 Родительская спора: {parent_spore.id}")
        evolution_print(f"   📍 Родительская позиция: {parent_spore.calc_2d_pos()}")
        evolution_print(f"   💰 Родительская стоимость: {parent_spore.logic.cost:.6f}")
        
        # Проверяем, может ли родительская спора продолжать эволюцию
        debug_output = self.config.get('trajectory_optimization', {}).get('debug_output', False)
        if not parent_spore.can_evolve():
            reason = ""
            if not parent_spore.is_alive():
                reason = f"💀 мертва (dt={parent_spore.logic.optimal_dt})"
            elif parent_spore.evolution_completed:
                reason = "🏁 эволюция завершена (объединение)"
            else:
                reason = "❓ неизвестная причина"
                
            evolution_print(f"   ❌ ОСТАНОВКА: Родительская спора не может создавать детей - {reason}")
            if debug_output:
                if not parent_spore.is_alive():
                    trajectory_print(f"💀 Спора {parent_spore.id} мертва и не может создавать детей")
                elif parent_spore.evolution_completed:
                    trajectory_print(f"🏁 Спора {parent_spore.id} завершила эволюцию и не может создавать детей")
            return None
            
        evolution_print(f"   ✅ Родительская спора может эволюционировать")
        evolution_print(f"   🎮 Использует управление: {parent_spore.logic.optimal_control}")
        evolution_print(f"   ⏱️  Использует dt: {parent_spore.logic.optimal_dt}")
            
        # ИСПРАВЛЕНИЕ: Сначала создаем новую спору (позволяем реально дойти до позиции)
        evolution_print(f"   🏗️  СОЗДАНИЕ новой споры...")
        new_spore = parent_spore.step(control=parent_spore.logic.optimal_control, dt=parent_spore.logic.optimal_dt)
        new_position_2d = new_spore.calc_2d_pos()
        
        evolution_print(f"   ✅ Новая спора создана:")
        evolution_print(f"      📍 Новая позиция: {new_position_2d}")
        evolution_print(f"      💰 Новая стоимость: {new_spore.logic.cost:.6f}")
        evolution_print(f"      📉 Изменение стоимости: {parent_spore.logic.cost - new_spore.logic.cost:.6f}")
        
        # Теперь проверяем, есть ли рядом с НОВОЙ СПОРОЙ другие существующие споры
        tolerance = self.config.get('trajectory_optimization', {}).get('trajectory_merge_tolerance', 0.05)
        trajectory_print(f"   🔍 ПРОВЕРКА близости (tolerance: {tolerance})...")
        
        existing_spore = self.find_nearby_spore(new_position_2d, tolerance, exclude_spore=parent_spore)
        
        # Отладочная информация (опциональная)
        if debug_output:
            trajectory_print(f"🔍 Проверка траектории после создания:")
            trajectory_print(f"   Родитель {parent_spore.id}: {parent_spore.calc_2d_pos()}")
            trajectory_print(f"   Новая спора позиция: {new_position_2d}")
            trajectory_print(f"   Tolerance: {tolerance}")
            trajectory_print(f"   Управление: {parent_spore.logic.optimal_control}")
            trajectory_print(f"   dt: {parent_spore.logic.optimal_dt}")
            if existing_spore:
                trajectory_print(f"   Найдена близкая спора {existing_spore.id} в позиции: {existing_spore.calc_2d_pos()}")
                distance = np.linalg.norm(new_position_2d - existing_spore.calc_2d_pos())
                trajectory_print(f"   Расстояние до неё: {distance:.4f}")
            else:
                trajectory_print(f"   Ближайших спор не найдено")
        
        if existing_spore is not None:
            # Найдена близкая спора - удаляем новую спору и создаем связь к существующей
            distance = np.linalg.norm(new_position_2d - existing_spore.calc_2d_pos())
            trajectory_print(f"   🎯 НАЙДЕНА близкая спора {existing_spore.id}:")
            trajectory_print(f"      📍 Позиция близкой: {existing_spore.calc_2d_pos()}")
            trajectory_print(f"      📏 Расстояние: {distance:.6f}")
            trajectory_print(f"      💰 Стоимость близкой: {existing_spore.logic.cost:.6f}")
            
            trajectory_print(f"   ♻️  УДАЛЕНИЕ только что созданной споры...")
            destroy(new_spore)  # Удаляем только что созданную спору
            
            trajectory_print(f"   🔗 СОЗДАНИЕ связи объединения...")
            self.create_link_to_existing(parent_spore, existing_spore)
            
            # Завершаем эволюцию родительской споры после объединения
            trajectory_print(f"   🏁 ЗАВЕРШЕНИЕ эволюции родительской споры...")
            parent_spore.mark_evolution_completed()
            
            trajectory_print(f"✅ ОБЪЕДИНЕНИЕ ЗАВЕРШЕНО:")
            trajectory_print(f"   🔄 Траектория: спора {parent_spore.id} → спора {existing_spore.id}")
            trajectory_print(f"   🏁 Спора {parent_spore.id} завершила эволюцию")
            trajectory_print(f"   ♻️  Новая спора удалена\n")
            
            if debug_output:
                trajectory_print(f"🔄 Траектория объединена: спора {parent_spore.id} → существующая спора {existing_spore.id}")
                trajectory_print(f"🏁 Спора {parent_spore.id} завершила эволюцию (объединение)")
                trajectory_print(f"♻️  Новая спора удалена, так как рядом есть существующая")
            return existing_spore
        
        # Близкая спора не найдена - добавляем новую спору как обычно
        evolution_print(f"   ✅ Близких спор не найдено - добавляем новую спору")
        evolution_print(f"   ➕ ДОБАВЛЕНИЕ в систему...")
        # ID будет присвоен в add_spore
        
        self.add_spore(new_spore)
        self.zoom_manager.register_object(new_spore, f"spore_{new_spore.id}")

        # Create a link if enabled in config
        if self.config.get('link', {}).get('show', True):
            debug_print(f"   🔗 СОЗДАНИЕ обычной связи: {parent_spore.id} → {new_spore.id}")
            new_link = Link(parent_spore,
                            new_spore,
                            color_manager=self.color_manager,
                            zoom_manager=self.zoom_manager,
                            config=self.config)
            self.links.append(new_link)
            self.zoom_manager.register_object(new_link, f"link_{len(self.links)}")

        # --- ИСПРАВЛЕНИЕ (финальная, простая версия): ---
        # Вызываем полный цикл обновления, как это происходит при зуме.
        # Это гарантирует, что все объекты, включая новый линк и его родителей,
        # будут в согласованном состоянии.
        self.zoom_manager.update_transform()

        evolution_print(f"✅ НОВАЯ СПОРА СОЗДАНА:")
        evolution_print(f"   🆔 ID: {new_spore.id}")
        evolution_print(f"   📍 Финальная позиция: {new_spore.calc_2d_pos()}")
        evolution_print(f"   💰 Финальная стоимость: {new_spore.logic.cost:.6f}")
        evolution_print(f"   🔗 Связей в системе: {len(self.links)}")
        evolution_print(f"   🔸 Спор в системе: {len(self.objects)}\n")

        return new_spore

    def generate_random_spore_in_spawn_area(self) -> Optional[Spore]:
        """Создает случайную спору в случайной позиции внутри spawn area."""
        if not self.spawn_area:
            always_print("⚠️ Spawn area не задан, невозможно создать случайную спору")
            return None
        
        # Получаем случайную 2D позицию внутри spawn area
        random_position_2d = self.spawn_area.sample_random_point()
        
        # Конвертируем в 3D позицию (Y=0 для плоскости XZ)
        random_position_3d = (random_position_2d[0], 0.0, random_position_2d[1])
        
        # Создаем новую спору в случайной позиции
        # Получаем параметры цели от любой существующей споры или используем defaults
        goal_position = None
        if self.objects:
            goal_position = self.objects[0].goal_position
        else:
            # Если нет спор, используем значения из конфига
            goal_position = self.config.get('spore', {}).get('goal_position', [3.14159, 0])
        
        # Создаем новую спору
        new_spore = Spore(
            pendulum=self.pendulum,
            dt=self.config.get('pendulum', {}).get('dt', 0.1),
            goal_position=goal_position,
            scale=self.config.get('spore', {}).get('scale', 0.05),
            position=random_position_3d,
            color_manager=self.color_manager,
            config=self.config.get('spore', {})
        )
        
        # Добавляем спору в менеджер 
        self.add_spore(new_spore)
        self.zoom_manager.register_object(new_spore, f"random_spore_{new_spore.id}")
        
        always_print(f"🎲 Создана случайная спора {new_spore.id} в позиции {random_position_2d}")
        
        return new_spore
    
    def generate_candidate_spores(self) -> None:
        """Генерирует кандидатские споры с помощью дисков Пуассона."""
        if not self.spawn_area:
            always_print("⚠️ Spawn area не задан, невозможно создать кандидатов")
            return
        
        # Очищаем старых кандидатов
        for candidate in self.candidate_spores:
            # Удаляем из zoom_manager перед удалением объекта
            if hasattr(candidate, 'id'):
                self.zoom_manager.unregister_object(candidate.id)
            destroy(candidate)
        self.candidate_spores = []
        
        try:
            # Генерируем позиции через диски Пуассона
            candidate_positions_2d = self.spawn_area.sample_poisson_disk(self.min_radius)
            
            if len(candidate_positions_2d) == 0:
                always_print(f"⚠️ Не удалось сгенерировать кандидатов с радиусом {self.min_radius}")
                return
            
            # Получаем параметры цели
            goal_position = None
            if self.objects:
                goal_position = self.objects[0].goal_position
            else:
                goal_position = self.config.get('spore', {}).get('goal_position', [3.14159, 0])
            
            # Создаем кандидатские споры
            for i, pos_2d in enumerate(candidate_positions_2d):
                pos_3d = (pos_2d[0], 0.0, pos_2d[1])
                
                candidate = Spore(
                    pendulum=self.pendulum,
                    dt=self.config.get('pendulum', {}).get('dt', 0.1),
                    goal_position=goal_position,
                    scale=self.config.get('spore', {}).get('scale', 0.05),
                    position=pos_3d,
                    color_manager=self.color_manager,
                    config=self.config.get('spore', {}),
                    is_ghost=False  # Кандидаты не призраки
                )
                
                candidate.id = f"candidate_{i}"
                candidate.is_candidate = True
                candidate.set_color_type('candidate')  # Белый полупрозрачный
                
                self.candidate_spores.append(candidate)
                self.zoom_manager.register_object(candidate, candidate.id)
            
            self.candidate_count = len(self.candidate_spores)
            candidate_print(f"🎯 Создано {self.candidate_count} кандидатов с радиусом {self.min_radius}")
            
        except Exception as e:
            always_print(f"❌ Ошибка при создании кандидатов: {e}")
    
    def activate_random_candidate(self) -> Optional[Spore]:
        """Выбирает случайного кандидата и делает его активной спорой."""
        if not self.candidate_spores:
            always_print("⚠️ Нет доступных кандидатов")
            return None
        
        candidate_print(f"\n🎲 АКТИВАЦИЯ СЛУЧАЙНОГО КАНДИДАТА:")
        candidate_print(f"   📊 Доступно кандидатов: {len(self.candidate_spores)}")
        
        # Выбираем случайного кандидата
        import random
        selected_candidate = random.choice(self.candidate_spores)
        
        candidate_print(f"   🎯 Выбран кандидат: {selected_candidate.id}")
        candidate_print(f"   📍 Позиция кандидата: {selected_candidate.calc_2d_pos()}")
        candidate_print(f"   💰 Стоимость кандидата: {selected_candidate.logic.cost:.6f}")
        
        # Сохраняем позицию и цель до удаления
        candidate_position = selected_candidate.real_position.copy()
        candidate_goal = selected_candidate.goal_position
        
        candidate_print(f"   🗑️  УДАЛЕНИЕ кандидата из системы...")
        
        # Удаляем кандидата из zoom_manager ПЕРЕД удалением объекта
        if hasattr(selected_candidate, 'id'):
            self.zoom_manager.unregister_object(selected_candidate.id)
        
        # Удаляем его из кандидатов
        self.candidate_spores.remove(selected_candidate)
        
        # Удаляем кандидата
        destroy(selected_candidate)
        
        # Обновляем счетчик
        self.candidate_count = len(self.candidate_spores)
        candidate_print(f"   📉 Осталось кандидатов: {self.candidate_count}")
        
        candidate_print(f"   🏗️  СОЗДАНИЕ активной споры на том же месте...")
        
        # Создаем новую активную спору в той же позиции
        new_spore = Spore(
            pendulum=self.pendulum,
            dt=self.config.get('pendulum', {}).get('dt', 0.1),
            goal_position=candidate_goal,
            scale=self.config.get('spore', {}).get('scale', 0.05),
            position=candidate_position,
            color_manager=self.color_manager,
            config=self.config.get('spore', {})
        )
        
        candidate_print(f"   ➕ ДОБАВЛЕНИЕ активной споры в систему...")
        
        # Добавляем новую спору через стандартную систему
        self.add_spore(new_spore)
        self.zoom_manager.register_object(new_spore, f"activated_spore_{new_spore.id}")
        
        candidate_print(f"✅ КАНДИДАТ АКТИВИРОВАН:")
        candidate_print(f"   🆔 Новый ID: {new_spore.id}")
        candidate_print(f"   📍 Позиция: {candidate_position}")
        candidate_print(f"   💰 Стоимость: {new_spore.logic.cost:.6f}")
        candidate_print(f"   🎮 Управление: {new_spore.logic.optimal_control}")
        candidate_print(f"   ⏱️  dt: {new_spore.logic.optimal_dt}")
        candidate_print(f"   💚 Жива: {new_spore.is_alive()}")
        candidate_print(f"   🚀 Может эволюционировать: {new_spore.can_evolve()}\n")
        
        return new_spore

    def evolve_all_candidates_to_completion(self) -> None:
        """
        Развивает всех кандидатов до смерти или остановки эволюции.
        
        Для каждого кандидата:
        1. Активирует его (превращает в обычную спору)
        2. Развивает до тех пор пока спора может эволюционировать
        3. Останавливается когда спора умирает или завершает эволюцию
        """
        if not self.candidate_spores:
            always_print("⚠️ Нет кандидатов для развития")
            return
        
        total_candidates = len(self.candidate_spores)
        always_print(f"\n🚀 НАЧИНАЕМ МАССОВОЕ РАЗВИТИЕ КАНДИДАТОВ:")
        always_print(f"   📊 Всего кандидатов: {total_candidates}")
        always_print(f"   🎯 Цель: развить каждого до смерти или остановки эволюции\n")
        
        processed_count = 0
        
        # Обрабатываем кандидатов пока они есть
        while self.candidate_spores:
            processed_count += 1
            remaining = len(self.candidate_spores)
            
            always_print(f"🔄 ОБРАБАТЫВАЕМ КАНДИДАТА {processed_count}/{total_candidates} (осталось: {remaining})")
            evolution_print(f"   📊 Детали: активируем кандидата для развития")
            
            # Активируем первого доступного кандидата
            activated_spore = self.activate_random_candidate()
            if not activated_spore:
                always_print("❌ Не удалось активировать кандидата")
                break
            
            evolution_print(f"✅ Кандидат активирован как спора {activated_spore.id}")
            evolution_print(f"   📍 Стартовая позиция: {activated_spore.calc_2d_pos()}")
            evolution_print(f"   💰 Стартовая стоимость: {activated_spore.logic.cost:.6f}")
            
            # Развиваем спору до конца
            evolution_step = 0
            max_steps = 100  # Защита от бесконечного цикла
            
            evolution_print(f"   🧬 НАЧИНАЕМ ЭВОЛЮЦИЮ...")
            
            while activated_spore.can_evolve() and evolution_step < max_steps:
                evolution_step += 1
                evolution_print(f"      📈 Шаг эволюции {evolution_step}:")
                evolution_print(f"         📍 Текущая позиция: {activated_spore.calc_2d_pos()}")
                evolution_print(f"         💰 Текущая стоимость: {activated_spore.logic.cost:.6f}")
                evolution_print(f"         💚 Жива: {activated_spore.is_alive()}")
                evolution_print(f"         🏁 Эволюция завершена: {activated_spore.evolution_completed}")
                
                # Создаем следующую спору
                result = self.generate_new_spore()
                
                if result is None:
                    evolution_print(f"         🛑 generate_new_spore вернул None - остановка")
                    break
                elif result == activated_spore:
                    evolution_print(f"         🔄 Спора осталась той же - продолжаем")
                    continue
                else:
                    evolution_print(f"         ➡️  Создана новая спора {result.id}")
                    activated_spore = result  # Переходим к новой споре
            
            # Причина завершения эволюции
            reason = ""
            if evolution_step >= max_steps:
                reason = f"🛑 достигнут лимит шагов ({max_steps})"
            elif not activated_spore.is_alive():
                reason = f"💀 спора умерла (dt = {activated_spore.logic.optimal_dt})"
            elif activated_spore.evolution_completed:
                reason = "🏁 эволюция завершена (объединение)"
            else:
                reason = "❓ неизвестная причина"
            
            evolution_print(f"   🎯 ЭВОЛЮЦИЯ ЗАВЕРШЕНА после {evolution_step} шагов:")
            evolution_print(f"      🔍 Причина: {reason}")
            evolution_print(f"      📍 Финальная позиция: {activated_spore.calc_2d_pos()}")
            evolution_print(f"      💰 Финальная стоимость: {activated_spore.logic.cost:.6f}")
            evolution_print(f"      💚 Финальное состояние: {'жива' if activated_spore.is_alive() else 'мертва'}")
            
            # Показываем прогресс завершения кандидата
            always_print(f"✅ КАНДИДАТ {processed_count}/{total_candidates} ЗАВЕРШЕН ({reason})")
            evolution_print()
        
        always_print(f"🎉 МАССОВОЕ РАЗВИТИЕ ЗАВЕРШЕНО!")
        always_print(f"   ✅ Обработано кандидатов: {processed_count}")
        always_print(f"   📊 Осталось кандидатов: {len(self.candidate_spores)}")
        always_print(f"   🔸 Всего активных спор в системе: {len(self.objects)}")
        always_print(f"   🔗 Всего связей: {len(self.links)}\n")
    
    def adjust_min_radius(self, multiplier: float) -> None:
        """Изменяет минимальный радиус мультипликативно и перегенерирует кандидатов."""
        old_radius = self.min_radius
        self.min_radius = max(0.05, self.min_radius * multiplier)  # Минимум 0.05
        
        if abs(self.min_radius - old_radius) > 0.001:  # Перегенерируем при любом значимом изменении
            candidate_print(f"🔧 Радиус изменен: {old_radius:.3f} → {self.min_radius:.3f} (×{multiplier:.2f})")
            self.generate_candidate_spores()
    
    def find_nearby_spore(self, position_2d: np.ndarray, tolerance: float = 0.1, exclude_spore: Optional[Spore] = None) -> Optional[Spore]:
        """
        Ищет спору в радиусе tolerance от заданной позиции.
        
        Args:
            position_2d: 2D позиция для поиска
            tolerance: Радиус поиска
            exclude_spore: Спора, которую нужно исключить из поиска (обычно родительская)
            
        Returns:
            Ближайшая спора в радиусе или None
        """
        min_distance = float('inf')
        closest_spore = None
        
        for spore in self.objects:
            # Пропускаем исключаемую спору
            if exclude_spore is not None and spore.id == exclude_spore.id:
                continue
                
            # Пропускаем только призраков (мертвые споры учитываем для объединения)
            if (hasattr(spore, 'is_ghost') and spore.is_ghost):
                continue
                
            spore_pos_2d = spore.calc_2d_pos()
            distance = np.linalg.norm(position_2d - spore_pos_2d)
            
            if distance <= tolerance and distance < min_distance:
                min_distance = distance
                closest_spore = spore
                
        return closest_spore
    
    def create_link_to_existing(self, from_spore: Spore, to_spore: Spore) -> None:
        """
        Создает связь между существующими спорами.
        
        Args:
            from_spore: Родительская спора (та, которая создала новую)
            to_spore: Целевая спора (существующая)
        """
        trajectory_print(f"   🔗 ДЕТАЛИ СОЗДАНИЯ СВЯЗИ ОБЪЕДИНЕНИЯ:")
        trajectory_print(f"      🏁 ОТ: спора {from_spore.id} (родительская)")
        trajectory_print(f"         📍 Позиция: {from_spore.calc_2d_pos()}")
        trajectory_print(f"         💰 Стоимость: {from_spore.logic.cost:.6f}")
        trajectory_print(f"         💚 Жива: {from_spore.is_alive()}")
        trajectory_print(f"      🎯 К: спора {to_spore.id} (существующая)")
        trajectory_print(f"         📍 Позиция: {to_spore.calc_2d_pos()}")
        trajectory_print(f"         💰 Стоимость: {to_spore.logic.cost:.6f}")
        trajectory_print(f"         💚 Жива: {to_spore.is_alive()}")
        
        if self.config.get('link', {}).get('show', True):
            # ИСПРАВЛЕНИЕ: Стрелка должна показывать что from_spore "приходит" к to_spore
            # Логика: траектория от родительской споры ведет к существующей споре
            trajectory_print(f"      🏹 Направление стрелки: {from_spore.id} → {to_spore.id}")
            
            new_link = Link(from_spore,  # parent_spore (откуда приходит траектория)
                           to_spore,     # existing_spore (куда приходит траектория)
                           color_manager=self.color_manager,
                           zoom_manager=self.zoom_manager,
                           config=self.config)
            # Выделяем особым цветом связи между существующими спорами
            link_color = self.color_manager.get_color('link', 'active')
            new_link.color = link_color
            trajectory_print(f"      🎨 Цвет связи: {link_color} (активная связь объединения)")
            
            self.links.append(new_link)
            self.zoom_manager.register_object(new_link, f"merge_link_{len(self.links)}")
            
            trajectory_print(f"      🔧 Обновление геометрии и трансформации...")
            # Обновляем геометрию и трансформацию
            new_link.update_geometry()
            self.zoom_manager.update_transform()
            
            trajectory_print(f"   ✅ СВЯЗЬ СОЗДАНА: спора {from_spore.id} → спора {to_spore.id} (направление объединения)")
            trajectory_print(f"      🔗 Всего связей в системе: {len(self.links)}")
        else:
            trajectory_print(f"      ❌ Создание связей отключено в конфигурации")

    def get_last_active_spore(self) -> Optional[Spore]:
        """Получить последнюю спору, способную к эволюции (не goal, не завершенная)"""
        for spore in reversed(self.objects):
            if not (hasattr(spore, 'is_goal') and spore.is_goal):
                # Возвращаем первую найденную не-цель (для обратной совместимости)
                # но приоритет отдаем спорам, способным к эволюции
                if hasattr(spore, 'can_evolve') and spore.can_evolve():
                    return spore
        
        # Если нет спор, способных к эволюции, возвращаем последнюю не-цель
        for spore in reversed(self.objects):
            if not (hasattr(spore, 'is_goal') and spore.is_goal):
                return spore
        return None

    def update_ghost_link(self) -> None:
        """Создает и обновляет связь от последней споры к оптимальной призрачной споре."""
        last_spore = self.get_last_active_spore()
        if not last_spore or not self.optimal_ghost_spore:
            return

        # print('--------------------------------')
        # print(f'last_spore: {last_spore}')
        # print(f'last_spore cost: {last_spore.logic.cost}')
        # print(f'optimal_control: {last_spore.logic.optimal_control}')
        # print('--------------------------------')
        
        # Создаем или обновляем связь к оптимальной призрачной споре
        if not self.ghost_link:
            self.ghost_link = Link(last_spore, self.optimal_ghost_spore, 
                                   color_manager=self.color_manager, 
                                   zoom_manager=self.zoom_manager,
                                   config=self.config)
            self.ghost_link.color = self.color_manager.get_color('link', 'ghost')
            self.zoom_manager.register_object(self.ghost_link, "ghost_link")
        else:
            # Обновляем связь если она уже существует
            self.ghost_link.parent_spore = last_spore
            self.ghost_link.child_spore = self.optimal_ghost_spore
            self.ghost_link.update_geometry()
        
        # Уведомляем AngelManager
        if self.angel_manager:
            self.angel_manager.update_ghost_link_angel(last_spore, self.optimal_ghost_spore)

    def create_ghost_spores(self, N: int) -> None:
        """Создает N визуализаторов предсказаний."""
        last_spore = self.get_last_active_spore()
        if not last_spore:
            return

        # Перед созданием новых очищаем старые
        for visualizer in self.prediction_visualizers:
            visualizer.destroy()
        self.prediction_visualizers = []

        # Создаем N новых визуализаторов
        for i in range(N):
            visualizer = PredictionVisualizer(
                parent_spore=last_spore,
                color_manager=self.color_manager,
                zoom_manager=self.zoom_manager,
                cost_function=self.angel_manager.cost_function if self.angel_manager else None,
                config=self.config,
                spore_id=f"ghost_pred_{i}"
            )
            self.prediction_visualizers.append(visualizer)

    def sample_ghost_spores(self, N: int = 4) -> None:
        """Создает специфичных призраков с конкретными управлениями и цветами."""
        # Всегда создаем 4 призрака: 2 красных (макс), 1 зеленый (ноль), 1 синий (опт)
        if not self.prediction_visualizers:
            self.create_ghost_spores(4)
        
        if not self.prediction_visualizers:
            return

        last_spore = self.get_last_active_spore()
        if not last_spore:
            return

        # Получаем границы управления
        min_control, max_control = last_spore.logic.pendulum.get_control_bounds()
        optimal_control = last_spore.logic.optimal_control[0] if last_spore.logic.optimal_control is not None else 0.0
        dt_to_use = last_spore.logic.optimal_dt if last_spore.logic.optimal_dt is not None else last_spore.logic.dt

        # Определяем конкретные управления и цвета
        ghost_configs = [
            {'control': max_control, 'color': 'ghost_max', 'name': 'max_pos'},     # Красный +макс
            {'control': min_control, 'color': 'ghost_max', 'name': 'max_neg'},     # Красный -макс  
            {'control': 0.0, 'color': 'ghost_zero', 'name': 'zero'},              # Зеленый
            {'control': optimal_control, 'color': 'ghost_optimal', 'name': 'optimal'}  # Синий
        ]

        # Симулируем состояния для каждого управления
        controls = [config['control'] for config in ghost_configs]
        states = last_spore.logic.simulate_controls(controls, dt_to_use)

        # Обновляем каждый визуализатор
        for i, visualizer in enumerate(self.prediction_visualizers):
            if i < len(states) and i < len(ghost_configs):
                visualizer.update(states[i])
                # Устанавливаем цвет в зависимости от типа управления
                if hasattr(visualizer, 'ghost_spore') and visualizer.ghost_spore:
                    visualizer.ghost_spore.set_color_type(ghost_configs[i]['color'])
                
                # Сохраняем ссылку на оптимальную призрачную спору (синюю)
                if ghost_configs[i]['name'] == 'optimal':
                    self.optimal_ghost_spore = visualizer.ghost_spore
                
                visualizer.set_visibility(True)
                # print(f"👻 Призрак {ghost_configs[i]['name']}: control={ghost_configs[i]['control']:.3f}, state={states[i]}")
            else:
                visualizer.set_visibility(False)




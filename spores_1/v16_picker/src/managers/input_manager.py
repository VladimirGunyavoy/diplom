from ursina import held_keys, mouse
import time
import numpy as np
from typing import Optional



# Предварительное объявление классов для аннотаций типов
# Это позволяет избежать циклических импортов
from ..visual.scene_setup import SceneSetup
from ..managers.zoom_manager import ZoomManager
from ..managers.spore_manager import SporeManager
from ..managers.spawn_area_manager import SpawnAreaManager
from ..managers.param_manager import ParamManager
from ..visual.ui_setup import UI_setup
from ..logic.tree import run_area_optimization
from .buffer_merge_manager import BufferMergeManager

# Forward declaration для ManualSporeManager
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..managers.manual_spore_manager import ManualSporeManager
    from ..managers.angel_manager import AngelManager
    from ..visual.cost_visualizer import CostVisualizer
    from ..managers.dt_manager import DTManager
    from ..visual.controls_window import ControlsWindow
    from ..managers.picker_manager import PickerManager
    
from ..utils.debug_output import always_print


class InputManager:
    """
    Централизованный класс для обработки всего пользовательского ввода.
    Принимает ссылки на все необходимые менеджеры и объекты сцены,
    и делегирует им команды в зависимости от нажатой клавиши.
    """
    def __init__(self,
                 scene_setup: Optional[SceneSetup] = None,
                 zoom_manager: Optional[ZoomManager] = None,
                 spore_manager: Optional[SporeManager] = None,
                 spawn_area_manager: Optional[SpawnAreaManager] = None,
                 param_manager: Optional[ParamManager] = None,
                 ui_setup: Optional[UI_setup] = None,
                 angel_manager: Optional['AngelManager'] = None,
                 cost_visualizer: Optional['CostVisualizer'] = None,
                 manual_spore_manager: Optional["ManualSporeManager"] = None,
                 dt_manager: Optional['DTManager'] = None,
                 picker_manager: Optional['PickerManager'] = None):
        
        self.scene_setup: Optional[SceneSetup] = scene_setup
        self.zoom_manager: Optional[ZoomManager] = zoom_manager
        self.spore_manager: Optional[SporeManager] = spore_manager
        self.spawn_area_manager: Optional[SpawnAreaManager] = spawn_area_manager
        self.param_manager: Optional[ParamManager] = param_manager
        self.ui_setup: Optional[UI_setup] = ui_setup
        self.angel_manager: Optional['AngelManager'] = angel_manager
        self.cost_visualizer: Optional['CostVisualizer'] = cost_visualizer
        self.manual_spore_manager: Optional["ManualSporeManager"] = manual_spore_manager
        self.dt_manager: Optional['DTManager'] = dt_manager
        self.picker_manager: Optional['PickerManager'] = picker_manager

        # 🔄 v16: BufferMergeManager для клавиши M
        self.buffer_merge_manager = BufferMergeManager(distance_threshold=1.5e-3)
        
        # 🆕 v16: Окно с инструкциями управления
        self.controls_window: Optional['ControlsWindow'] = None

        print(f"[IM] constructed, dt_manager id={id(self.dt_manager) if self.dt_manager else None}")

        # 🔍 Флаг для включения детальной отладки призрачного дерева
        self.debug_ghost_tree = False

        # Настройки для генерации спор по клавише 'f'
        self.f_key_down_time: float = 0
        self.long_press_threshold: float = 0.4
        self.spawn_interval: float = 0.1
        self.next_spawn_time: float = 0  # Время следующего спавна
        
        # v13_manual: отслеживание кликов мыши
        self.previous_mouse_left = False

        # Подписка на изменения dt (чтобы призраки и линки всегда реагировали)
        if self.dt_manager and hasattr(self.dt_manager, "subscribe_on_change"):
            self.dt_manager.subscribe_on_change(self._on_dt_changed)
            print("[IM] subscribed to DTManager.on_change")

        # 🆕 v16: Командная система
        self._setup_command_system()
        
        # === ОТЛАДКА ИНИЦИАЛИЗАЦИИ ===
        print(f"🔧 [DEBUG] InputManager инициализирован:")
        print(f"   📹 zoom_manager: {self.zoom_manager}")
        print(f"   📹 zoom_manager is None: {self.zoom_manager is None}")
        print(f"   🎮 scene_setup: {self.scene_setup}")
        print(f"   🌱 spore_manager: {self.spore_manager}")
        # === КОНЕЦ ОТЛАДКИ ===

    def _setup_command_system(self):
        """Настраивает централизованную командную систему."""
        self.commands = {
            # === SPORES === (removed - outdated functionality)
            
            # === ZOOM ===
            'e': {
                'description': 'zoom in',
                'handler': self._handle_zoom_in,
                'category': 'zoom',
                'enabled': lambda: self.zoom_manager is not None
            },
            't': {
                'description': 'zoom out',
                'handler': self._handle_zoom_out,
                'category': 'zoom',
                'enabled': lambda: self.zoom_manager is not None
            },
            'r': {
                'description': 'reset zoom',
                'handler': self._handle_zoom_reset,
                'category': 'zoom',
                'enabled': lambda: self.zoom_manager is not None
            },
            '1': {
                'description': 'scale down',
                'handler': self._handle_spores_scale_down,
                'category': 'zoom',
                'enabled': lambda: self.zoom_manager is not None
            },
            '2': {
                'description': 'scale up',
                'handler': self._handle_spores_scale_up,
                'category': 'zoom',
                'enabled': lambda: self.zoom_manager is not None
            },
            
            # === CANDIDATES ===
            '5': {
                'description': 'candidates -',
                'handler': self._handle_candidates_radius_down,
                'category': 'candidates',
                'enabled': lambda: self.spore_manager is not None
            },
            '6': {
                'description': 'candidates +',
                'handler': self._handle_candidates_radius_up,
                'category': 'candidates',
                'enabled': lambda: self.spore_manager is not None
            },
            
            # === VISUAL ===
            'y': {
                'description': 'toggle angels',
                'handler': self._handle_toggle_angels,
                'category': 'visual',
                'enabled': lambda: self.angel_manager is not None
            },
            'u': {
                'description': 'toggle frame',
                'handler': self._handle_toggle_frame,
                'category': 'visual',
                'enabled': lambda: self.scene_setup is not None
            },
            
            # === TIME ===
            ',': {
                'description': 'reset dt (comma)',
                'handler': self._handle_dt_reset,
                'category': 'time',
                'enabled': lambda: self.dt_manager is not None
            },
            'm': {
                'description': 'merge ghost tree',
                'handler': self._handle_merge_optimization,
                'category': 'merge',
                'enabled': lambda: self.manual_spore_manager is not None
            },
            'j': {
                'description': 'dt stats',
                'handler': self._handle_dt_stats,
                'category': 'time',
                'enabled': lambda: self.dt_manager is not None
            },
            
            # === OPTIMIZE ===
            'o': {
                'description': 'optimize tree',
                'handler': self._handle_optimization,
                'category': 'optimize',
                'enabled': lambda: self.manual_spore_manager is not None and self.dt_manager is not None
            },
            'p': {
                'description': 'apply pairs',
                'handler': self._handle_optimal_pairs,
                'category': 'optimize',
                'enabled': lambda: self.manual_spore_manager is not None
            },
            
            # === TREE ===  
            'k': {
                'description': 'toggle mode',
                'handler': self._handle_toggle_creation_mode,
                'category': 'tree',
                'enabled': lambda: self.manual_spore_manager is not None
            },
            '7': {
                'description': 'depth 1',
                'handler': self._handle_tree_depth_1,
                'category': 'tree',
                'enabled': lambda: self._is_tree_mode()
            },
            '8': {
                'description': 'depth 2',
                'handler': self._handle_tree_depth_2,
                'category': 'tree',
                'enabled': lambda: self._is_tree_mode()
            },
            
            # === GHOSTS ===
            ';': {
                'description': 'toggle ghosts', 
                'handler': self._handle_toggle_ghosts,
                'category': 'ghosts',
                'enabled': lambda: self.manual_spore_manager is not None
            },
            
            # === DEBUG ===
            # 'l': {
            #     'description': 'plot tree graph (debug)',
            #     'handler': self._handle_debug_plot_tree,
            #     'category': 'debug',
            #     'enabled': lambda: self.manual_spore_manager is not None
            # },
            'l': {
                'description': 'graph stats',
                'handler': self._handle_all_graph_stats,
                'category': 'debug',
                'enabled': lambda: self.spore_manager is not None
            },
            'v': {
                'description': 'valence analysis',
                'handler': self._handle_valence_analysis,
                'category': 'debug',
                'enabled': lambda: self.spore_manager is not None
            },
            
            
            # === DELETE ===
            'ctrl+c': {
                'description': 'clear all',
                'handler': self._handle_clear_all,
                'category': 'delete',
                'enabled': lambda: self.manual_spore_manager is not None
            },
            'z': {
                'description': 'delete last',
                'handler': self._handle_delete_last_group,
                'category': 'delete',
                'enabled': lambda: self.manual_spore_manager is not None
            },
            'i': {
                'description': 'groups stats',
                'handler': self._handle_groups_stats,
                'category': 'stats',
                'enabled': lambda: self.manual_spore_manager is not None
            },
            # Команды движения камеры (w, a, s, d, space, shift) обрабатываются в first person controller
            
                                            # === КУРСОР ===
                'alt': {
                    'description': 'toggle cursor',
                    'handler': self._handle_toggle_cursor,
                    'category': 'ui',
                    'enabled': lambda: self.scene_setup is not None
                },
            
            # === MOUSE WHEEL ===
            'scroll up': {
                'description': 'zoom in',
                'handler': self._handle_scroll_up,
                'category': 'zoom',
                'enabled': lambda: self.zoom_manager is not None
            },
            'scroll down': {
                'description': 'zoom out',
                'handler': self._handle_scroll_down,
                'category': 'zoom',
                'enabled': lambda: self.zoom_manager is not None
            },
            
            # === DIAGNOSTIC ===
            
            # === UI ===
        }
        
        print(f"✅ Командная система настроена: {len(self.commands)} команд")

    def get_commands_by_category(self) -> dict:
        """Группирует команды по категориям для генерации справки."""
        categories = {}
        for key, cmd_info in self.commands.items():
            category = cmd_info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((key, cmd_info['description']))
        return categories

    def print_commands_help(self):
        """Выводит справку по всем доступным командам."""
        print("\n📋 СПРАВКА ПО КОМАНДАМ:")
        print("=" * 50)
        
        categories = self.get_commands_by_category()
        for category, commands in categories.items():
            print(f"\n🎯 {category.upper()}:")
            for key, desc in commands:
                enabled = self.commands[key]['enabled']()
                status = "✅" if enabled else "❌" 
                print(f"   {status} {key.upper()}: {desc}")
                
        print("\n💡 Легенда: ✅ - доступно, ❌ - менеджер отключен")
        print("=" * 50)
        
        # Добавляем список всех занятых клавиш
        print("\n🎹 ЗАНЯТЫЕ КЛАВИШИ:")
        print("=" * 30)
        
        # Разбиваем клавиши на категории
        digits = []
        letters = []
        combinations = []
        
        for key in self.commands.keys():
            if key.isdigit():
                digits.append(key.upper())
            elif len(key) == 1 and key.isalpha():
                letters.append(key.upper())
            else:
                combinations.append(key.upper())
        
        # Выводим по категориям
        if digits:
            print(f"   ЦИФРЫ: {', '.join(sorted(digits))}")
        if letters:
            print(f"   БУКВЫ: {', '.join(sorted(letters))}")
        if combinations:
            print(f"   КОМБИНАЦИИ: {', '.join(sorted(combinations))}")
        
        # Подсчитываем общее количество
        total_keys = len(self.commands)
        print(f"\n📊 Всего занято клавиш: {total_keys}")
        print("=" * 30)

    def get_free_keys(self) -> list:
        """Возвращает список свободных клавиш."""
        all_keys = set('abcdefghijklmnopqrstuvwxyz1234567890')
        used_keys = set(self.commands.keys())
        return sorted(list(all_keys - used_keys))

    def _is_tree_mode(self) -> bool:
        """Проверяет активен ли режим дерева."""
        return (self.manual_spore_manager is not None and 
                hasattr(self.manual_spore_manager, 'creation_mode') and
                self.manual_spore_manager.creation_mode == 'tree')

    def update(self) -> None:
        """
        Этот метод должен вызываться каждый кадр для обработки непрерывного ввода.
        """
        # 🔄 v16: обработка кликов мыши для материализации буферного графа
        if self.manual_spore_manager:
            current_mouse_left = mouse.left
            # Обнаруживаем нажатие ЛКМ (переход с False на True)
            if current_mouse_left and not self.previous_mouse_left:
                
                # Проверяем есть ли данные в буферном графе
                if self.buffer_merge_manager.has_buffer_data():
                    print(f"   🖱️ ЛКМ: Материализация буферного графа в реальный...")
                    
                    # 🔍 ДИАГНОСТИКА БУФЕРНЫХ ДАННЫХ
                    buffer_positions = getattr(self.buffer_merge_manager, 'buffer_positions', {})
                    buffer_links = getattr(self.buffer_merge_manager, 'buffer_links', [])
                    print(f"   📊 Буферных спор: {len(buffer_positions)}")
                    print(f"   🔗 Буферных связей: {len(buffer_links)}")
                    
                    if len(buffer_links) == 0:
                        print(f"   ⚠️ ВНИМАНИЕ: Буферных связей нет! Материализация создаст споры без связей.")
                    
                    # Материализуем буферный граф
                    result = self.buffer_merge_manager.materialize_buffer_to_real(
                        spore_manager=self.spore_manager,
                        zoom_manager=self.zoom_manager,
                        color_manager=self.spore_manager.color_manager if self.spore_manager else None,
                        pendulum=self.manual_spore_manager.deps.pendulum if hasattr(self.manual_spore_manager, 'deps') else None,
                        config=self.manual_spore_manager.deps.config if hasattr(self.manual_spore_manager, 'deps') else {}
                    )
                    
                    if result['success']:
                        stats = result['stats']
                        print(f"   ✅ Материализация завершена: {stats['spores_created']} спор, {stats['links_created']} связей")
                        
                        # ✅ Материализация работает, добавление в историю происходит в BufferMergeManager
                        print(f"   📚 Материализованные споры автоматически добавлены в историю групп")
                    else:
                        print(f"   ❌ Ошибка материализации: {result.get('error', 'Неизвестная ошибка')}")
                        
                else:
                    # Fallback: создаем споры как раньше
                    print(f"   🖱️ ЛКМ: Буферный граф пуст, создание спор...")
                    created_spores = self.manual_spore_manager.create_spore_at_cursor()
                    if created_spores:
                        print(f"   ✅ Создано {len(created_spores)} спор (1 родитель + 2 ребёнка + 2 линка)")
                        
            self.previous_mouse_left = current_mouse_left
        
        # Логика для непрерывной генерации спор при удержании 'f'
        if held_keys['f']:  # type: ignore
            if self.f_key_down_time > 0:  # Проверяем, что нажатие было зафиксировано
                now = time.time()
                if now >= self.next_spawn_time:
                    if self.spore_manager:
                        self.spore_manager.generate_new_spore()
                    # Устанавливаем время для следующей генерации
                    self.next_spawn_time = now + self.spawn_interval
        else:
            # Сбрасываем таймер если клавиша не нажата
            self.f_key_down_time = 0
        
        # Обработка движения камеры через held_keys
        if self.scene_setup and self.scene_setup.player:
            step = self.scene_setup.base_speed * 0.016  # Примерно 60 FPS
            
            # Движение вверх (Space)
            if held_keys['space']:  # type: ignore
                self.scene_setup.player.y += step
            
            # Движение вниз (Shift)
            if held_keys['shift']:  # type: ignore
                self.scene_setup.player.y -= step

    # === ОБРАБОТЧИКИ КОМАНД ===
    
    def _handle_spore_creation(self):
        """Обработчик создания новой споры (F)."""
        if self.spore_manager:
            self.spore_manager.generate_new_spore()
        # Запускаем таймеры для возможной непрерывной генерации
        now = time.time()
        self.f_key_down_time = now
        self.next_spawn_time = now + self.long_press_threshold

    def _handle_candidate_activation(self):
        """Обработчик активации кандидата (G)."""
        if self.spore_manager:
            self.spore_manager.activate_random_candidate()

    def _handle_evolve_all(self):
        """Обработчик эволюции всех кандидатов (V)."""
        if self.spore_manager:
            self.spore_manager.evolve_all_candidates_to_completion()

    def _handle_zoom_in(self):
        """Обработчик приближения камеры (E)."""
        if self.zoom_manager:
            self.zoom_manager.zoom_in()

    def _handle_zoom_out(self):
        """Обработчик отдаления камеры (T)."""
        if self.zoom_manager:
            self.zoom_manager.zoom_out()

    def _handle_zoom_reset(self):
        """Обработчик сброса зума (R)."""
        if self.zoom_manager:
            self.zoom_manager.reset_zoom()

    def _handle_spores_scale_down(self):
        """Обработчик уменьшения масштаба спор (1)."""
        if self.zoom_manager:
            self.zoom_manager.decrease_spores_scale()

    def _handle_spores_scale_up(self):
        """Обработчик увеличения масштаба спор (2)."""
        if self.zoom_manager:
            self.zoom_manager.increase_spores_scale()

    def _handle_candidates_radius_down(self):
        """Обработчик уменьшения радиуса кандидатов (5)."""
        if self.spore_manager:
            self.spore_manager.adjust_min_radius(1/1.2)

    def _handle_candidates_radius_up(self):
        """Обработчик увеличения радиуса кандидатов (6)."""
        if self.spore_manager:
            self.spore_manager.adjust_min_radius(1.2)

    def _handle_toggle_angels(self):
        """Обработчик переключения ангелов (Y)."""
        if self.angel_manager:
            self.angel_manager.toggle_angels()

    def _handle_toggle_frame(self):
        """Обработчик переключения системы координат (U)."""
        if self.scene_setup and hasattr(self.scene_setup, 'frame'):
            self.scene_setup.frame.toggle_visibility()

    def _handle_dt_reset(self):
        """Обработчик сброса dt (M).""" 
        # Комплексный сброс как в оригинале
        if self.dt_manager:
            self.dt_manager.reset_dt()
        
        if self.manual_spore_manager:
            # Получаем текущий dt после сброса  
            current_dt = self.dt_manager.get_dt() if self.dt_manager else 0.001
            factor = 0.05
            if hasattr(self.manual_spore_manager, 'deps'):
                factor = self.manual_spore_manager.deps.config.get('tree', {}).get('dt_grandchildren_factor', 0.05)
            
            # Формируем стандартный dt_vector
            dt_children = np.array([+current_dt, -current_dt, +current_dt, -current_dt], dtype=float)
            base_gc = current_dt * factor
            dt_grandchildren = np.array([
                +base_gc, -base_gc, +base_gc, -base_gc,
                +base_gc, -base_gc, +base_gc, -base_gc
            ], dtype=float)
            
            standard_dt_vector = np.concatenate([dt_children, dt_grandchildren])
            self.manual_spore_manager.ghost_tree_dt_vector = standard_dt_vector
            
            # Обновляем предсказания
            if hasattr(self.manual_spore_manager, 'prediction_manager'):
                self.manual_spore_manager.prediction_manager.clear_predictions()
                self.manual_spore_manager._update_predictions()
            
            print(f"🔄 Все dt сброшены к стандартным значениям")

    def _handle_merge_optimization(self):
        """Обработчик мерджа призрачного дерева (M)."""
        print("[IM][M] Клавиша M нажата - запуск нового алгоритма мерджа!")
        
        try:
            # Получаем tree_logic из manual_spore_manager
            if not self.manual_spore_manager:
                print("❌ Manual spore manager не найден")
                return
                
            # Получаем последнее созданное дерево
            tree_logic = getattr(self.manual_spore_manager, '_last_tree_logic', None)
            if not tree_logic:
                print("❌ Призрачное дерево не найдено")
                print("   💡 Создайте дерево призраков сначала (наведите курсор на область)")
                return
                
            print("✅ Найдено призрачное дерево:")
            print(f"   📍 Корень: {tree_logic.root['position'] if tree_logic.root else 'нет'}")
            print(f"   👶 Детей: {len(tree_logic.children) if hasattr(tree_logic, 'children') else 0}")
            grandchildren_count = (len(tree_logic.grandchildren) 
                                 if hasattr(tree_logic, 'grandchildren') else 0)
            print(f"   👶👶 Внуков: {grandchildren_count}")
            
            # Передаем ссылку на manual_spore_manager для исследования
            self.buffer_merge_manager._manual_spore_manager_ref = self.manual_spore_manager
            
            # Запускаем новый алгоритм мерджа
            result = self.buffer_merge_manager.merge_ghost_tree(tree_logic, save_image=True)
            
            if result['success']:
                print("✅ Мердж успешно завершен!")
                
                # Выводим краткую статистику
                stats = result['stats']
                print(f"📊 Результат: {stats['added_to_buffer']} спор в буфере, "
                      f"{stats['merged_to_existing']} объединений")
                
                if 'image_path' in stats:
                    print(f"🖼️ Картинка сохранена: {stats['image_path']}")
                    
            else:
                print(f"❌ Ошибка мерджа: {result.get('error', 'Неизвестная ошибка')}")
                
        except Exception as e:
            print(f"❌ Исключение при мердже: {e}")
            import traceback
            traceback.print_exc()

    def _handle_standard_reset(self):
        """Обработчик полного сброса к стандартным dt ([)."""
        print(f"[IM][[]] Полный сброс к стандартному режиму...")
        
        # 1. Сбрасываем общий dt через dt_manager (как клавиша M)
        if self.dt_manager:
            self.dt_manager.reset_dt()
        
        # 2. Сбрасываем ghost_tree_dt_vector к стандартным значениям
        if self.manual_spore_manager:
            # Получаем текущий dt после сброса
            current_dt = self.dt_manager.get_dt() if self.dt_manager else 0.001
            
            # Получаем фактор для внуков из конфигурации
            factor = 0.05  # дефолтное значение (внуки в 20 раз меньше)
            if hasattr(self.manual_spore_manager, 'deps'):
                factor = self.manual_spore_manager.deps.config.get('tree', {}).get('dt_grandchildren_factor', 0.05)
            
            # Формируем стандартный dt_vector: 4 детей + 8 внуков
            dt_children = np.array([+current_dt, -current_dt, +current_dt, -current_dt], dtype=float)
            base_gc = current_dt * factor
            dt_grandchildren = np.array([
                +base_gc, -base_gc, +base_gc, -base_gc,
                +base_gc, -base_gc, +base_gc, -base_gc
            ], dtype=float)
            
            standard_dt_vector = np.concatenate([dt_children, dt_grandchildren])
            
            # Устанавливаем стандартный вектор
            self.manual_spore_manager.ghost_tree_dt_vector = standard_dt_vector
            
            # Сбрасываем baseline для корректного масштабирования
            self.manual_spore_manager.ghost_dt_baseline = current_dt
            
            # Обновляем предсказания
            if hasattr(self.manual_spore_manager, 'prediction_manager'):
                self.manual_spore_manager.prediction_manager.clear_predictions()
                self.manual_spore_manager._update_predictions()
            
            print(f"🔄 Все dt сброшены к стандартным значениям:")
            print(f"   📊 Общий dt: {current_dt}")
            print(f"   📊 Фактор внуков: {factor}")
            print(f"   📊 dt детей: {dt_children}")
            print(f"   📊 dt внуков: {dt_grandchildren}")
            print(f"   💡 Теперь внуки в {1/factor:.0f} раз меньше детей")
        else:
            print(f"⚠️ Manual spore manager не найден")

    def _handle_dt_stats(self):
        """Обработчик статистики dt (J)."""
        if self.dt_manager:
            self.dt_manager.print_stats()

    def _handle_optimal_pairs(self):
        """Обработчик оптимальных пар (P)."""
        if self.manual_spore_manager:
            self._apply_optimal_pairs_to_ghost_tree()

    def _handle_toggle_creation_mode(self):
        """Обработчик переключения режима создания (K)."""
        if self.manual_spore_manager and hasattr(self.manual_spore_manager, 'toggle_creation_mode'):
            self.manual_spore_manager.toggle_creation_mode()

    def _handle_tree_depth_1(self):
        """Обработчик установки глубины дерева = 1 (7)."""
        if self.manual_spore_manager and hasattr(self.manual_spore_manager, 'set_tree_depth'):
            self.manual_spore_manager.set_tree_depth(1)

    def _handle_tree_depth_2(self):
        """Обработчик установки глубины дерева = 2 (8).""" 
        if self.manual_spore_manager and hasattr(self.manual_spore_manager, 'set_tree_depth'):
            self.manual_spore_manager.set_tree_depth(2)

    def _handle_toggle_ghosts(self):
        """🆕 Обработчик переключения призрачной системы (:)."""
        if self.manual_spore_manager and hasattr(self.manual_spore_manager, 'toggle_ghost_system'):
            new_state = self.manual_spore_manager.toggle_ghost_system()
            if new_state:
                print("   💡 Подсказка: призраки снова следуют за курсором")
            else:
                print("   💡 Подсказка: теперь виден только чистый курсор")
        else:
            print("⚠️ ManualSporeManager или метод toggle_ghost_system не найден")

    def _handle_debug_toggle(self):
        """Обработчик переключения отладки (H)."""
        self.debug_ghost_tree = not self.debug_ghost_tree
        status = "ВКЛЮЧЕНА" if self.debug_ghost_tree else "ОТКЛЮЧЕНА"
        print(f"🔍 Детальная отладка призрачного дерева: {status}")

    def _handle_debug_plot_tree(self):
        """Обработчик отладочного графика дерева (L)."""
        if not self.manual_spore_manager:
            print("❌ Manual spore manager не найден")
            return
            
        try:
            # Пытаемся получить последнее созданное дерево
            tree_creation_manager = self.manual_spore_manager.tree_creation_manager
            
            # Проверяем есть ли у manual_spore_manager сохраненное дерево логики
            if hasattr(self.manual_spore_manager, '_last_tree_logic') and self.manual_spore_manager._last_tree_logic:
                tree_logic = self.manual_spore_manager._last_tree_logic
                print(f"🎯 Строим график последнего дерева...")
                
                # Вызываем debug метод
                save_path = tree_logic.debug_plot_tree()
                print(f"✅ График дерева сохранен: {save_path}")
                
            else:
                print("❌ Нет сохраненного дерева для отладки")
                print("💡 Создайте дерево через ЛКМ, затем нажмите L")
                
        except Exception as e:
            print(f"❌ Ошибка создания графика дерева: {e}")
            import traceback
            traceback.print_exc()

    # def _handle_help(self):  # Убрано - теперь используется окно управления
    #     """Обработчик вывода справки (N)."""
    #     self.print_commands_help()

    def _handle_optimization(self):
        """Обработчик оптимизации дерева (O)."""
        print(f"[IM][O] Клавиша O нажата!")
        try:
            print("[IM][O] Запуск оптимизации площади...")

            # ==== Достаём зависимости ====
            # pendulum находится в deps
            pendulum = getattr(self.manual_spore_manager, "deps", None)
            if pendulum:
                pendulum = getattr(pendulum, "pendulum", None)

            # dt-manager
            dt_manager = self.dt_manager

            print(f"[IM][O] Диагностика зависимостей:")
            print(f"   pendulum: {pendulum is not None} ({type(pendulum) if pendulum else 'None'})")
            print(f"   dt_manager: {dt_manager is not None} ({type(dt_manager) if dt_manager else 'None'})")

            if pendulum is None:
                raise RuntimeError("Не найден объект pendulum в manual_spore_manager.deps.pendulum")
            if dt_manager is None:
                raise RuntimeError("Не найден dt_manager")

            # ==== Создаём временное дерево и пары ====
            print("[IM][O] Создание временного дерева для оптимизации...")
            
            # Получаем позицию курсора
            mouse_pos = self.manual_spore_manager.get_mouse_world_position()
            if mouse_pos is None:
                raise RuntimeError("Не удалось получить позицию курсора")
            
            cursor_position_2d = np.array([mouse_pos[0], mouse_pos[1]])
            print(f"[IM][O] Позиция курсора: {cursor_position_2d}")
                
            # Импортируем нужные классы
            from ..logic.tree.spore_tree import SporeTree
            from ..logic.tree.spore_tree_config import SporeTreeConfig
            from ..logic.tree.pairs.find_optimal_pairs import find_optimal_pairs
            
            # Загружаем конфигурацию спаривания
            from ..logic.tree.tree_area_bridge import _load_pairing_config
            pairing_config = _load_pairing_config()
            
            # Получаем текущий dt из системы
            dt = dt_manager.get_dt() if dt_manager else 0.05
            
            # Создаем временное дерево для поиска пар
            tree_config = SporeTreeConfig(
                initial_position=cursor_position_2d,
                dt_base=dt,
                dt_grandchildren_factor=pairing_config.get('dt_grandchildren_factor', 0.2),
                show_debug=pairing_config.get('show_debug', True)
            )
            
            temp_tree = SporeTree(
                pendulum=pendulum,
                config=tree_config,
                auto_create=True  # Создает полное дерево автоматически
            )
            
            print(f"[IM][O] Временное дерево создано: {len(temp_tree.children)} детей, {len(temp_tree.grandchildren)} внуков")
            
            # Ищем оптимальные пары
            pairs = find_optimal_pairs(temp_tree, show=True)
            
            if pairs is None:
                raise RuntimeError("Не удалось найти пары")
            
            print(f"[IM][O] Найдено {len(pairs)} оптимальных пар")

            # ==== Вызов оптимизации ====
            # Параметры загружаются из config/json/config.json
            result = run_area_optimization(
                tree=temp_tree,
                pairs=pairs,
                pendulum=pendulum,
                dt_manager=dt_manager,
                # dt_bounds, optimization_method, max_iterations загружаются из конфига
            )

            # ==== Красивые логи (научная нотация) ====
            if result is None:
                print(f"[IM][O] ❌ Оптимизация вернула None - возможно, произошла ошибка")
                return
                
            try:
                start_area = result.get("start_area", result.get("initial_area", None))
                best_area  = result.get("best_area",  result.get("optimized_area", None))
                if start_area is not None and best_area is not None:
                    delta = best_area - start_area
                    rel = (delta / max(abs(start_area), 1e-12)) * 100.0
                    print(f"[IM][O] Готово. Площадь: {start_area:.6e} → {best_area:.6e} (Δ={delta:+.6e}, {rel:+.3f}%)")
                else:
                    print(f"[IM][O] Результат получен, но площади не найдены в result")
            except Exception as log_error:
                print(f"[IM][O] Ошибка в логах результата: {log_error}")
                
            print(f"[IM][O] Оптимизация завершена: {result}")
            
            # ==== ПРИМЕНЕНИЕ РЕЗУЛЬТАТОВ К ПРИЗРАЧНОМУ ДЕРЕВУ ====
            if result and result.get('success', False):
                try:
                    # Используем тот же алгоритм что и в _apply_optimal_pairs_to_ghost_tree()
                    print(f"[IM][O] Применяем оптимизированные dt к призрачному дереву...")
                    
                    # Получаем optimized_tree из результата
                    optimized_tree = result.get('optimized_tree')
                    if optimized_tree is None:
                        print(f"[IM][O] ⚠️ optimized_tree не найдено в результате")
                        return
                    
                    # Извлекаем dt из оптимизированного дерева (аналогично P)
                    # Инициализируем массивы исходными значениями
                    dt_children = np.array([child['dt'] for child in optimized_tree.children])
                    dt_grandchildren = np.array([gc['dt'] for gc in optimized_tree.grandchildren])
                    
                    print(f"[IM][O] Оптимизированные dt из дерева:")
                    print(f"   Дети: {dt_children}")
                    print(f"   Внуки: {dt_grandchildren}")
                    
                    # 🔍 ВАЖНО: Дети НЕ ОПТИМИЗИРУЮТСЯ - их dt остаются стандартными!
                    print(f"[IM][O] ⚠️  ВНИМАНИЕ: dt детей НЕ изменяются алгоритмом оптимизации!")
                    print(f"   Дети используют стандартные dt: {dt_children}")
                    print(f"   Только внуки получают оптимизированные dt")
                    
                    # Формируем финальный dt_vector из 12 элементов
                    # Первые 4 - dt детей, следующие 8 - dt внуков
                    dt_vector = np.concatenate([dt_children, dt_grandchildren])
                    
                    print(f"[IM][O] 🎯 Сформированный dt_vector:")
                    print(f"   dt_children (0:4): {dt_vector[:4]}")  
                    print(f"   dt_grandchildren (4:12): {dt_vector[4:12]}")
                    
                    # Подставляем в призрачное дерево
                    self.manual_spore_manager.ghost_tree_dt_vector = dt_vector

                    # Запомним базовый dt на момент оптимизации — нужен для масштабирования
                    if self.dt_manager:
                        self.manual_spore_manager.ghost_dt_baseline = self.dt_manager.get_dt()
                    
                    # Обновляем предсказания с новыми dt
                    if hasattr(self.manual_spore_manager, 'prediction_manager'):
                        self.manual_spore_manager.prediction_manager.clear_predictions()
                        self.manual_spore_manager._update_predictions()
                        print(f"[IM][O] ✅ Призрачное дерево обновлено с оптимизированными dt!")
                    else:
                        print(f"[IM][O] ⚠️ prediction_manager не найден")
                        
                except Exception as apply_error:
                    print(f"[IM][O] ❌ Ошибка применения результатов: {apply_error}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[IM][O] ⚠️ Оптимизация не удалась, результаты не применяются")
            
        except Exception as e:
            print(f"❌ [IM][O] Ошибка при оптимизации: {e}")
            import traceback
            traceback.print_exc()

    # Методы движения камеры удалены - обрабатываются в first person controller

    def _handle_toggle_cursor(self):
        """Обработчик переключения захвата курсора (Alt)."""
        # Alt обрабатывается в SceneSetup.toggle_freeze() через глобальную функцию input()
        # Этот обработчик не используется, но команда добавлена для справки
        pass

    def _handle_scroll_up(self):
        """Обработчик колесика мыши вверх (приближение)."""
        if self.zoom_manager:
            self.zoom_manager.zoom_in()

    def _handle_scroll_down(self):
        """Обработчик колесика мыши вниз (отдаление)."""
        if self.zoom_manager:
            self.zoom_manager.zoom_out()

    def _handle_clear_all(self):
        """Обработчик полной очистки всех спор (Ctrl+C)."""
        print("🧹 Клавиша Ctrl+C: Полная очистка всех спор и объектов")
        
        # Диагностика: Проверяем есть ли manual_spore_manager
        if self.manual_spore_manager:
            print(f"   🔍 ManualSporeManager найден: {type(self.manual_spore_manager)}")
            if hasattr(self.manual_spore_manager, 'clear_all'):
                print(f"   🔍 Метод clear_all найден")
                # Показываем статистику до очистки
                if hasattr(self.manual_spore_manager, 'created_links'):
                    print(f"   📊 created_links до очистки: {len(self.manual_spore_manager.created_links)}")
                if hasattr(self.manual_spore_manager, 'spore_groups_history'):
                    print(f"   📚 групп в истории: {len(self.manual_spore_manager.spore_groups_history)}")
                
                self.manual_spore_manager.clear_all()
            else:
                print(f"   ❌ Метод clear_all НЕ найден!")
        else:
            print(f"   ❌ ManualSporeManager НЕ найден!")
        
        # Очищаем через SporeManager
        if self.spore_manager:
            self.spore_manager.clear_all_manual()
        
        # 🧹 Очищаем буферный граф при полной очистке
        if hasattr(self, 'buffer_merge_manager'):
            clear_result = self.buffer_merge_manager.clear_buffer_graph()
            if clear_result['success']:
                print(f"   🧹 Буферный граф очищен: {clear_result['cleared_spores']} спор")

        # 🔄 Обновляем визуализацию (пустой граф)
        if hasattr(self.buffer_merge_manager, '_create_real_graph_visualization') and self.spore_manager:
            viz_path = self.buffer_merge_manager._create_real_graph_visualization(self.spore_manager)
            if viz_path:
                print(f"   🖼️ Обновлена визуализация: {viz_path}")

    def _handle_delete_last_group(self):
        """Обработчик удаления последней группы спор (Z)."""
        if self.manual_spore_manager:
            print("🗑️ Клавиша Z: Удаление последней группы спор")
            success = self.manual_spore_manager.delete_last_spore_group()
            if success:
                print("   ✅ Последняя группа спор успешно удалена")
                
                # 🔧 ПРИНУДИТЕЛЬНОЕ ОБНОВЛЕНИЕ PNG ПОСЛЕ УДАЛЕНИЯ
                try:
                    print("   🖼️ Принудительное обновление PNG...")
                    
                    # Добавляем задержку для завершения всех операций удаления
                    import time
                    time.sleep(0.1)
                    
                    # Обновляем визуализацию
                    if hasattr(self.buffer_merge_manager, '_create_real_graph_visualization') and self.spore_manager:
                        viz_path = self.buffer_merge_manager._create_real_graph_visualization(self.spore_manager)
                        if viz_path:
                            print(f"   ✅ PNG обновлен после удаления: {viz_path}")
                        else:
                            print("   ⚠️ Не удалось обновить PNG")
                    
                    # Диагностика: показываем сколько связей осталось
                    remaining_links = len(self.spore_manager.links) if self.spore_manager else 0
                    print(f"   📊 Связей осталось в SporeManager: {remaining_links}")
                    
                except Exception as e:
                    print(f"   ⚠️ Ошибка принудительного обновления PNG: {e}")
            else:
                print("   ❌ Не удалось удалить группу (возможно, нет групп для удаления)")
        else:
            print("   ❌ ManualSporeManager не найден!")

    def _handle_groups_stats(self):
        """Обработчик статистики истории групп (I)."""
        if self.manual_spore_manager:
            if hasattr(self.manual_spore_manager, 'get_groups_history_stats'):
                stats = self.manual_spore_manager.get_groups_history_stats()
                print("📊 Статистика истории групп:")
                print(f"   📚 Всего групп создано: {stats.get('total_groups', 0)}")
                print(f"   🔗 Всего линков создано: {stats.get('total_links', 0)}")
                print(f"   📋 Групп в истории: {len(self.manual_spore_manager.spore_groups_history)}")
                
                # Показываем последние несколько групп
                history = getattr(self.manual_spore_manager, 'spore_groups_history', [])
                if history:
                    print(f"   📖 Последние группы:")
                    for i, group in enumerate(history[-3:], start=len(history)-2):
                        if i > 0:
                            print(f"      Группа #{i}: {len(group)} спор")
            else:
                # Fallback статистика
                history = getattr(self.manual_spore_manager, 'spore_groups_history', [])
                links = getattr(self.manual_spore_manager, 'created_links', [])
                print("📊 Статистика истории групп:")
                print(f"   📚 Групп в истории: {len(history)}")
                print(f"   🔗 Активных линков: {len(links)}")
        else:
            print("   ❌ ManualSporeManager не найден!")

    def handle_input(self, key: str) -> None:
        """
        Новый централизованный обработчик команд через командную систему.
        """
        
        # Фильтруем события типа 'left alt up', 'right shift down', 'control' и т.д.
        # НО НЕ фильтруем scroll события
        if key == 'control' or (' ' in key and any(direction in key.lower() for direction in ['up', 'down', 'left', 'right']) and not key.startswith('scroll')):
            return  # Игнорируем события нажатия/отпускания модификаторов
        
        # === ОБРАБОТКА CTRL КОМБИНАЦИЙ ===
        ctrl_pressed = held_keys['left control'] or held_keys['right control']
        
        if ctrl_pressed and self.dt_manager:
            if key == 'e' or key in ['scroll up', 'wheel up', 'mouse wheel up']:
                self.dt_manager.increase_dt()
                return  # Не даём провалиться в обычный зум
            elif key == 't' or key in ['scroll down', 'wheel down', 'mouse wheel down']:
                self.dt_manager.decrease_dt()
                return  # Не даём провалиться в обычный зум
        
        # === КОНЕЦ ОБРАБОТКИ CTRL КОМБИНАЦИЙ ===
        
        # Проверяем есть ли команда для данной клавиши
        if key in self.commands:
            cmd_info = self.commands[key]
            
            # Проверяем доступность команды
            if cmd_info['enabled']():
                try:
                    cmd_info['handler']()
                except Exception as e:
                    print(f"❌ Ошибка выполнения команды '{key}': {e}")
            else:
                print(f"⚠️ Команда '{key}' недоступна (enabled вернул False)")
            return
        
        # Команды движения камеры (space hold, shift hold и т.д.) обрабатываются в first person controller
        
        # Обработка специальной команды Ctrl+C через held_keys (совместимость)
        elif held_keys['c'] and held_keys['left control'] and self.spore_manager:  # type: ignore
            # Вызываем обработчик из словаря
            if 'ctrl+c' in self.commands:
                self.commands['ctrl+c']['handler']()
            return
        
        # Ctrl+I removed - not working properly
        
        # Свободные клавиши
        elif key in ['x']:  # Убираем z, c, i из свободных
            print(f"🔓 Клавиша '{key}' свободна")
            return
            
        # Неизвестная команда - игнорируем
        else:
            pass  # Игнорируем неизвестные команды

    def _apply_optimal_pairs_to_ghost_tree(self) -> None:
        """
        Находит оптимальные пары для призрачного дерева под курсором
        и подставляет соответствующие dt в ghost_tree_dt_vector
        """
        try:
            # Проверяем что мы в режиме дерева глубины 2
            if not self.manual_spore_manager:
                print(f"❌ ManualSporeManager не найден")
                return
                
            if self.manual_spore_manager.creation_mode != 'tree' or self.manual_spore_manager.tree_depth != 2:
                print(f"❌ Оптимизация пар доступна только для деревьев глубины 2")
                print(f"   Текущий режим: {self.manual_spore_manager.creation_mode}")
                print(f"   Текущая глубина: {self.manual_spore_manager.tree_depth}")
                return
            
            # Получаем позицию курсора
            mouse_pos = self.manual_spore_manager.get_mouse_world_position()
            if mouse_pos is None:
                print("❌ Не удалось получить позицию курсора")
                return

            # Конвертируем в numpy array
            cursor_position_2d = np.array([mouse_pos[0], mouse_pos[1]])
            
            print(f"🎯 Поиск оптимальных пар для позиции {cursor_position_2d}")
            
            # 🔍 ДИАГНОСТИКА ДО СПАРИВАНИЯ: Показываем текущее состояние дерева
            if self.debug_ghost_tree:
                print(f"\n🔍 ДИАГНОСТИКА ДО СПАРИВАНИЯ:")
                
                # Получаем текущее призрачное дерево ДО изменений
                if hasattr(self.manual_spore_manager, 'prediction_manager') and self.manual_spore_manager.prediction_manager:
                    pred_manager = self.manual_spore_manager.prediction_manager
                    
                    # Показываем текущий dt_vector
                    if hasattr(self.manual_spore_manager, 'ghost_tree_dt_vector') and self.manual_spore_manager.ghost_tree_dt_vector is not None:
                        current_dt_vector = self.manual_spore_manager.ghost_tree_dt_vector
                        print(f"   📊 Текущий ghost_tree_dt_vector:")
                        print(f"      Дети (0:4): {current_dt_vector[:4]}")
                        print(f"      Внуки (4:12): {current_dt_vector[4:12]}")
                        
                        # 🔍 Сохраняем dt детей ДО для сравнения
                        self._children_dt_before = current_dt_vector[:4].copy()
                        print(f"   💾 Сохранили dt детей ДО: {self._children_dt_before}")
                    else:
                        print(f"   📊 Текущий ghost_tree_dt_vector: НЕ УСТАНОВЛЕН")
                        self._children_dt_before = None
                    
                    # Показываем количество призраков
                    if hasattr(pred_manager, 'prediction_visualizers'):
                        ghost_count = len([v for v in pred_manager.prediction_visualizers if v.ghost_spore])
                        print(f"   👻 Призрачных спор: {ghost_count}")
                    
                    # Показываем количество линков
                    if hasattr(pred_manager, 'prediction_links'):
                        print(f"   🔗 Призрачных линков: {len(pred_manager.prediction_links)}")
                    
                    # Показываем позиции всех призраков
                    if hasattr(pred_manager, 'prediction_visualizers'):
                        print(f"   📍 Позиции призраков ДО:")
                        for i, viz in enumerate(pred_manager.prediction_visualizers):
                            if viz.ghost_spore:
                                pos = (viz.ghost_spore.x, viz.ghost_spore.z)
                                print(f"      Призрак {i}: {pos}")
                        
                        # 🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА ВИЗУАЛЬНЫХ ОБЪЕКТОВ ДО
                        print(f"   🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА ВИЗУАЛЬНЫХ ОБЪЕКТОВ ДО:")
                        for i, viz in enumerate(pred_manager.prediction_visualizers):
                            if viz.ghost_spore:
                                print(f"      Призрак {i}:")
                                print(f"         ID: {viz.id}")
                                print(f"         Позиция: ({viz.ghost_spore.x:.6f}, {viz.ghost_spore.z:.6f})")
                                print(f"         real_position: {viz.ghost_spore.real_position}")
                                print(f"         enabled: {viz.ghost_spore.enabled}")
                                if hasattr(viz.ghost_spore, 'logic'):
                                    print(f"         logic.position_2d: {viz.ghost_spore.logic.position_2d}")
                                print(f"         Тип: {type(viz.ghost_spore)}")
                else:
                    print(f"   ❌ PredictionManager не найден")
                
                print(f"🔍 ДИАГНОСТИКА ДО СПАРИВАНИЯ ЗАВЕРШЕНА\n")
            
            # Импортируем нужные классы
            from ..logic.tree.spore_tree import SporeTree
            from ..logic.tree.spore_tree_config import SporeTreeConfig
            from ..logic.tree.pairs.find_optimal_pairs import find_optimal_pairs
            
            # Загружаем конфигурацию спаривания
            from ..logic.tree.tree_area_bridge import _load_pairing_config
            pairing_config = _load_pairing_config()
            
            # Получаем текущий dt из системы
            dt = self.dt_manager.get_dt() if self.dt_manager else 0.05
            
            # Создаем временное дерево для поиска пар
            tree_config = SporeTreeConfig(
                initial_position=cursor_position_2d,
                dt_base=dt,
                dt_grandchildren_factor=pairing_config.get('dt_grandchildren_factor', 0.2),
                show_debug=pairing_config.get('show_debug', True),
                skip_auto_merge=True  # НОВОЕ: отключаем автоматическое объединение для временных деревьев
            )
            
            temp_tree = SporeTree(
                pendulum=self.manual_spore_manager.deps.pendulum,
                config=tree_config,
                auto_create=True  # Создает полное дерево автоматически
            )
            
            print(f"🌲 Временное дерево создано: {len(temp_tree.children)} детей, {len(temp_tree.grandchildren)} внуков")
            
            # Ищем оптимальные пары
            pairs = find_optimal_pairs(temp_tree, show=True)
            
            if pairs is None:
                print(f"❌ Не удалось найти пары")
                return
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: должно быть ровно 4 пары
            if len(pairs) != 4:
                raise Exception(f"ОШИБКА: Найдено {len(pairs)} пар, а должно быть ровно 4!")
            
            print(f"✅ Найдено {len(pairs)} оптимальных пар")
            
            # Извлекаем dt из пар
            # Инициализируем массивы исходными значениями
            dt_children = np.array([child['dt'] for child in temp_tree.children])
            dt_grandchildren = np.array([gc['dt'] for gc in temp_tree.grandchildren])
            
            print(f"📊 Исходные dt детей: {dt_children}")
            print(f"📊 Исходные dt внуков: {dt_grandchildren}")
            
            # 🔍 ВАЖНО: Дети НЕ ОПТИМИЗИРУЮТСЯ - их dt остаются стандартными!
            print(f"⚠️  ВНИМАНИЕ: dt детей НЕ изменяются алгоритмом спаривания!")
            print(f"   Дети используют стандартные dt: {dt_children}")
            print(f"   Только внуки получают оптимизированные dt из пар")
            
            # Обновляем dt внуков согласно найденным парам
            for pair_idx, (gc_i, gc_j, meeting_info) in enumerate(pairs):
                # Извлекаем оптимальные времена (ВАЖНО: сохраняем знаки!)
                optimal_dt_i = meeting_info['time_gc']      # время для gc_i
                optimal_dt_j = meeting_info['time_partner'] # время для gc_j
                
                # Обновляем массив dt_grandchildren
                dt_grandchildren[gc_i] = optimal_dt_i
                dt_grandchildren[gc_j] = optimal_dt_j
                
                print(f"   Пара {pair_idx+1}: gc_{gc_i} ↔ gc_{gc_j}")
                print(f"     gc_{gc_i}: {temp_tree.grandchildren[gc_i]['dt']:+.6f} → {optimal_dt_i:+.6f}")
                print(f"     gc_{gc_j}: {temp_tree.grandchildren[gc_j]['dt']:+.6f} → {optimal_dt_j:+.6f}")
                print(f"     Расстояние встречи: {meeting_info['distance']:.6f}")
            
            # Формируем финальный dt_vector из 12 элементов
            # Первые 4 - dt детей, следующие 8 - dt внуков
            dt_vector = np.concatenate([dt_children, dt_grandchildren])
            
            print(f"🎯 Сформированный dt_vector:")
            print(f"   dt_children (0:4): {dt_vector[:4]}")  
            print(f"   dt_grandchildren (4:12): {dt_vector[4:12]}")
            
            # Подставляем в призрачное дерево
            self.manual_spore_manager.ghost_tree_dt_vector = dt_vector

            # --- зафиксируем baseline для последующего масштабирования колесиком
            if self.manual_spore_manager is not None and self.dt_manager is not None:
                try:
                    cur_dt = float(self.dt_manager.get_dt())
                except Exception:
                    # fallback: возьмём из конфига, если есть
                    cur_dt = float(getattr(self.dt_manager, "current_dt", 0.1))
                self.manual_spore_manager.ghost_dt_baseline = cur_dt
                print(f"[IM] ghost_dt_baseline set to {cur_dt:.6f} after pairing")

            # Запомним базовый dt на момент вычисления пар — нужен для масштабирования
            if self.dt_manager:
                self.manual_spore_manager.ghost_dt_baseline = self.dt_manager.get_dt()
            
            # ВАЖНО: Принудительно обновляем предсказания чтобы показать спаренное дерево
            # Принудительно обновляем призрачное дерево с новым dt_vector
            # Очищаем старые предсказания и пересоздаем с новым dt_vector
            
            if self.debug_ghost_tree:
                print(f"🔄 Начинаем обновление призрачного дерева...")
                print(f"   📊 ghost_tree_dt_vector установлен: {dt_vector is not None}")
                print(f"   📊 Длина dt_vector: {len(dt_vector) if dt_vector is not None else 'None'}")
                
                # Передаем флаг отладки в prediction_manager
                self.manual_spore_manager.prediction_manager.debug_ghost_tree = True
                
                self.manual_spore_manager.prediction_manager.clear_predictions()
                print(f"   🧹 Предсказания очищены")
                
                self.manual_spore_manager._update_predictions()
                print(f"   🔄 _update_predictions() вызван")
                
                # Отключаем отладку после использования
                self.manual_spore_manager.prediction_manager.debug_ghost_tree = False
            else:
                self.manual_spore_manager.prediction_manager.clear_predictions()
                self.manual_spore_manager._update_predictions()

            print(f"🔄 Призрачное дерево пересоздано с новыми dt")
            
            print(f"✅ Призрачное дерево обновлено со спаренными dt!")
            
            # --- гарантируем подписку на изменение dt после спаривания
            if self.dt_manager and hasattr(self.dt_manager, "subscribe_on_change"):
                try:
                    self.dt_manager.subscribe_on_change(self._on_dt_changed)
                    print(f"[IM] re-subscribed to DTManager.on_change after pairing (DT id={id(self.dt_manager)})")
                    # Диагностика: покажем подписчиков
                    if hasattr(self.dt_manager, "debug_subscribers"):
                        self.dt_manager.debug_subscribers()
                except Exception as ex:
                    print(f"[IM] failed to re-subscribe after pairing: {ex}")
            
            # Сразу привести всё в соответствие текущему dt и лимитам длины
            if self.dt_manager and hasattr(self, "_on_dt_changed"):
                self._on_dt_changed()
            
            # Жёсткая синхронизация лимитов длины через PredictionManager
            if self.manual_spore_manager and hasattr(self.manual_spore_manager, "prediction_manager"):
                pm = self.manual_spore_manager.prediction_manager
                if hasattr(pm, "update_links_max_length"):
                    max_len = self.dt_manager.get_max_link_length() if self.dt_manager else None
                    print(f"[IM] Жёсткая синхронизация: applying max_len={max_len} to {len(getattr(pm,'prediction_links',[]))} prediction_links")
                    pm.update_links_max_length(max_len)
            
            # 🔍 ДИАГНОСТИКА ПОСЛЕ СПАРИВАНИЯ: Показываем состояние обновленного дерева
            if self.debug_ghost_tree:
                print(f"\n🔍 ДИАГНОСТИКА ПОСЛЕ СПАРИВАНИЯ:")
                
                # Получаем текущее призрачное дерево
                if hasattr(self.manual_spore_manager, 'prediction_manager') and self.manual_spore_manager.prediction_manager:
                    pred_manager = self.manual_spore_manager.prediction_manager
                    
                    # Показываем dt_vector
                    if hasattr(self.manual_spore_manager, 'ghost_tree_dt_vector') and self.manual_spore_manager.ghost_tree_dt_vector is not None:
                        dt_vector = self.manual_spore_manager.ghost_tree_dt_vector
                        print(f"   📊 ghost_tree_dt_vector:")
                        print(f"      Дети (0:4): {dt_vector[:4]} (НЕ ИЗМЕНЕНЫ - стандартные)")
                        print(f"      Внуки (4:12): {dt_vector[4:12]} (ОПТИМИЗИРОВАНЫ)")
                        
                        # 🔍 ПОДТВЕРЖДЕНИЕ: Дети действительно не изменились
                        print(f"   🔍 АНАЛИЗ dt детей:")
                        for i, dt in enumerate(dt_vector[:4]):
                            print(f"      Ребенок {i}: dt = {dt:+.6f} (стандартный)")
                        
                        # 🔍 СРАВНЕНИЕ dt детей ДО и ПОСЛЕ
                        if hasattr(self, '_children_dt_before') and self._children_dt_before is not None:
                            print(f"   🔍 СРАВНЕНИЕ dt детей:")
                            for i, (dt_before, dt_after) in enumerate(zip(self._children_dt_before, dt_vector[:4])):
                                if abs(dt_before - dt_after) < 1e-10:
                                    print(f"      Ребенок {i}: dt = {dt_before:+.6f} → {dt_after:+.6f} ✅ НЕ ИЗМЕНИЛСЯ")
                                else:
                                    print(f"      Ребенок {i}: dt = {dt_before:+.6f} → {dt_after:+.6f} 🔄 ИЗМЕНИЛСЯ!")
                        else:
                            print(f"   🔍 СРАВНЕНИЕ dt детей: Нет данных ДО для сравнения")
                    
                    # Показываем количество призраков
                    if hasattr(pred_manager, 'prediction_visualizers'):
                        ghost_count = len([v for v in pred_manager.prediction_visualizers if v.ghost_spore])
                        print(f"   👻 Призрачных спор: {ghost_count}")
                    
                    # Показываем количество линков
                    if hasattr(pred_manager, 'prediction_links'):
                        print(f"   🔗 Призрачных линков: {len(pred_manager.prediction_links)}")
                    
                    # Показываем позиции всех призраков
                    if hasattr(pred_manager, 'prediction_visualizers'):
                        print(f"   📍 Позиции призраков:")
                        for i, viz in enumerate(pred_manager.prediction_visualizers):
                            if viz.ghost_spore:
                                pos = (viz.ghost_spore.x, viz.ghost_spore.z)
                                print(f"      Призрак {i}: {pos}")
                        
                        # 🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА ВИЗУАЛЬНЫХ ОБЪЕКТОВ
                        print(f"   🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА ВИЗУАЛЬНЫХ ОБЪЕКТОВ:")
                        for i, viz in enumerate(pred_manager.prediction_visualizers):
                            if viz.ghost_spore:
                                print(f"      Призрак {i}:")
                                print(f"         ID: {viz.id}")
                                print(f"         Позиция: ({viz.ghost_spore.x:.6f}, {viz.ghost_spore.z:.6f})")
                                print(f"         real_position: {viz.ghost_spore.real_position}")
                                print(f"         enabled: {viz.ghost_spore.enabled}")
                                if hasattr(viz.ghost_spore, 'logic'):
                                    print(f"         logic.position_2d: {viz.ghost_spore.logic.position_2d}")
                                print(f"         Тип: {type(viz.ghost_spore)}")
                else:
                    print(f"   ❌ PredictionManager не найден")
                
                print(f"🔍 ДИАГНОСТИКА ЗАВЕРШЕНА\n")
            
        except Exception as e:
            print(f"❌ Ошибка применения пар: {e}")
            import traceback
            traceback.print_exc()

    def _on_dt_changed(self):
        """
        Глобальная реакция на изменение dt:
        - СНАЧАЛА масштабируем ghost_tree_dt_vector (если он есть),
        - потом пересобираем призрачное дерево,
        - применяем лимит длины ко всем линкам.
        """
        try:
            # 0) СНАЧАЛА: масштабируем dt-вектор спаренного дерева под новый dt
            if (self.manual_spore_manager 
                and getattr(self.manual_spore_manager, "ghost_tree_dt_vector", None) is not None
                and getattr(self.manual_spore_manager, "ghost_dt_baseline", None) is not None
                and self.dt_manager):
                
                import numpy as np
                old_vec = self.manual_spore_manager.ghost_tree_dt_vector
                old_base = float(self.manual_spore_manager.ghost_dt_baseline)
                new_base = float(self.dt_manager.get_dt())
                print(f"[InputManager._on_dt_changed] scale ghost dt-vector: baseline {old_base:.6f} -> {new_base:.6f}")

                if old_base > 0:
                    k = new_base / old_base
                    before_sample = old_vec.copy()
                    scaled = np.sign(old_vec) * (np.abs(old_vec) * k)
                    self.manual_spore_manager.ghost_tree_dt_vector = scaled
                    self.manual_spore_manager.ghost_dt_baseline = new_base

                    # Диагностика: покажем пару первых элементов ДО/ПОСЛЕ
                    def _fmt(arr):
                        return "[" + ", ".join(f"{x:+.6f}" for x in arr[:4]) + (" ..." if len(arr) > 4 else "") + "]"
                    print(f"[InputManager._on_dt_changed] k={k:.6f}")
                    print(f"[InputManager._on_dt_changed] dt_vec before: {_fmt(before_sample)}")
                    print(f"[InputManager._on_dt_changed] dt_vec after : {_fmt(scaled)}")
                else:
                    print("[InputManager._on_dt_changed] WARNING: ghost_dt_baseline == 0, skip scaling.")

            # 1) Пересобрать призрачные предсказания (дети/внуки двигаются)
            if self.manual_spore_manager and hasattr(self.manual_spore_manager, "_update_predictions"):
                self.manual_spore_manager.prediction_manager.debug_ghost_tree = True  # временно включим лог
                self.manual_spore_manager._update_predictions()
                self.manual_spore_manager.prediction_manager.debug_ghost_tree = False

            # 2) Применить новый лимит длины для ВСЕХ линков
            max_len = None
            if self.dt_manager and hasattr(self.dt_manager, "get_max_link_length"):
                max_len = self.dt_manager.get_max_link_length()

            # 2a) Обычные линки (через SporeManager)
            if self.spore_manager and hasattr(self.spore_manager, "update_links_max_length"):
                self.spore_manager.update_links_max_length(max_len)

            # 2b) Призрачные линки (через PredictionManager)
            if self.manual_spore_manager and hasattr(self.manual_spore_manager, "prediction_manager"):
                pm = self.manual_spore_manager.prediction_manager
                if hasattr(pm, "update_links_max_length"):
                    pm.update_links_max_length(max_len)
                # На всякий случай обновим геометрию каждого призрачного линка
                if hasattr(pm, "prediction_links"):
                    for link in pm.prediction_links:
                        try:
                            link.update_geometry()
                        except Exception:
                            pass

            # 3) Обновим общую трансформацию зума, чтобы всё подхватилось
            if self.zoom_manager and hasattr(self.zoom_manager, "update_transform"):
                self.zoom_manager.update_transform()

        except Exception as e:
            print(f"[InputManager] _on_dt_changed error: {e}")

    def _handle_merge_grandchildren(self):
        """Обработчик объединения внуков и копирования призрачной структуры (M)."""
        try:
            print("\n" + "="*60)
            print("🔄 ЗАПУСК ОПЕРАЦИИ МЕРДЖА С ИСПОЛЬЗОВАНИЕМ ГРАФОВ")
            print("="*60)
            
            if not self.manual_spore_manager:
                print("❌ ManualSporeManager не найден")
                return
                
            # Получаем призрачный граф из PredictionManager
            prediction_manager = getattr(self.manual_spore_manager, 'prediction_manager', None)
            if not prediction_manager:
                print("❌ PredictionManager не найден")
                return
                
            # Показываем статистику ДО операции
            print("📊 СТАТИСТИКА ДО МЕРДЖА:")
            if hasattr(prediction_manager, 'get_ghost_graph_stats'):
                prediction_manager.get_ghost_graph_stats()
            if self.spore_manager and hasattr(self.spore_manager, 'get_graph_stats'):
                self.spore_manager.get_graph_stats()
                
            # Выполняем объединение внуков (существующая логика)
            tree_creation_manager = getattr(self.manual_spore_manager, 'tree_creation_manager', None)
            if not tree_creation_manager:
                print("❌ TreeCreationManager не найден")
                return
                
            # Получаем логику дерева
            tree_logic = getattr(tree_creation_manager, '_last_tree_logic', None) or \
                        getattr(self.manual_spore_manager, '_last_tree_logic', None)
            
            if not tree_logic:
                print("❌ Логика дерева не найдена. Создайте дерево призраков сначала.")
                return
                
            if not hasattr(tree_logic, 'grandchildren') or len(tree_logic.grandchildren) == 0:
                print("❌ Список внуков пуст")
                return
                
            # Показываем состояние ДО объединения
            print(f"📊 ДО объединения: {len(tree_logic.grandchildren)} внуков")
            
            # Вызываем объединение
            merge_result = tree_logic.merge_close_grandchildren(distance_threshold=1e-2)
            
            if merge_result['total_merged'] > 0:
                print(f"✅ Объединено {merge_result['total_merged']} пар внуков")
                print(f"📊 ПОСЛЕ объединения: {merge_result['remaining_grandchildren']} внуков")
                
                # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Пересинхронизируем призрачный граф
                print("\n🔄 СИНХРОНИЗАЦИЯ ПРИЗРАЧНОГО ГРАФА С ОБЪЕДИНЕНИЯМИ:")
                if hasattr(prediction_manager, 'clear_predictions'):
                    prediction_manager.clear_predictions()
                    print("   🧹 Очищены старые призрачные предсказания")

                # Пересоздаем призрачные предсказания с объединенной структурой
                if hasattr(self.manual_spore_manager, '_update_predictions'):
                    self.manual_spore_manager._update_predictions()
                    print("   🔄 Пересозданы призрачные предсказания с объединениями")
                else:
                    print("   ⚠️ Метод _update_predictions не найден")
                
                # НОВОЕ: Отладочная визуализация призрачного графа
                if (hasattr(prediction_manager, 'ghost_graph') and 
                    hasattr(prediction_manager.ghost_graph, 'create_debug_visualization')):
                    
                    print("\n🔍 СОЗДАНИЕ ОТЛАДОЧНОЙ ВИЗУАЛИЗАЦИИ ПРИЗРАЧНОГО ГРАФА:")
                    ghost_viz_path = prediction_manager.ghost_graph.create_debug_visualization("ghost_graph_debug")
                    
                    if ghost_viz_path:
                        print(f"👁️ Откройте файл для анализа: {ghost_viz_path}")
                    
                    # Выводим структуру призрачного графа в консоль
                    if hasattr(prediction_manager.ghost_graph, 'print_graph_structure'):
                        prediction_manager.ghost_graph.print_graph_structure()
                    
                    # Также создаем визуализацию реального графа для сравнения
                    if (self.spore_manager and hasattr(self.spore_manager, 'graph') and
                        hasattr(self.spore_manager.graph, 'create_debug_visualization')):
                        real_viz_path = self.spore_manager.graph.create_debug_visualization("real_graph_debug")
                        if real_viz_path:
                            print(f"👁️ Реальный граф для сравнения: {real_viz_path}")
                            
                        # Выводим структуру реального графа
                        if hasattr(self.spore_manager.graph, 'print_graph_structure'):
                            self.spore_manager.graph.print_graph_structure()
                    
                else:
                    print("❌ Призрачный граф или метод визуализации не найдены")
                
                # 🔧 ПРИНУДИТЕЛЬНАЯ ОЧИСТКА И ПЕРЕСОЗДАНИЕ ПРИЗРАЧНОГО ГРАФА
                print("\n🔧 ПРИНУДИТЕЛЬНАЯ ОЧИСТКА ПРИЗРАЧНОГО ГРАФА:")
                if hasattr(prediction_manager, 'ghost_graph') and prediction_manager.ghost_graph:
                    old_nodes = len(prediction_manager.ghost_graph.nodes)
                    prediction_manager.ghost_graph.clear()
                    print(f"   🧹 Очищен призрачный граф: {old_nodes} узлов удалено")

                # Очищаем визуальные предсказания
                if hasattr(prediction_manager, 'clear_predictions'):
                    prediction_manager.clear_predictions()
                    print("   🧹 Очищены визуальные предсказания")

                # Принудительно пересоздаем призрачные предсказания с объединенной структурой
                if hasattr(self.manual_spore_manager, '_update_predictions'):
                    print("   🔄 Пересоздаем призрачные предсказания...")
                    self.manual_spore_manager._update_predictions()
                    
                    # Проверяем результат
                    if hasattr(prediction_manager, 'ghost_graph') and prediction_manager.ghost_graph:
                        new_nodes = len(prediction_manager.ghost_graph.nodes)
                        print(f"   ✅ Новый призрачный граф: {new_nodes} узлов")
                        # Ожидаем: 1 корень + 4 ребенка + 4 объединенных внука = 9 узлов
                        expected = 9
                        if new_nodes == expected:
                            print(f"   ✅ Количество узлов корректно (ожидалось {expected})")
                        else:
                            print(f"   ⚠️ Неожиданное количество узлов: {new_nodes} (ожидалось {expected})")
                    else:
                        print("   ❌ Призрачный граф не был пересоздан")
                else:
                    print("   ❌ Метод _update_predictions не найден")

                print("🔄 Призрачные предсказания принудительно обновлены")
                
            else:
                print("📊 Объединение не требуется - все внуки достаточно далеко (> 1e-2)")
                
            # 🔍 ПРОВЕРКА ДУБЛИРОВАНИЙ В ПРИЗРАЧНОМ ГРАФЕ:
            print("\n🔍 ПРОВЕРКА ДУБЛИРОВАНИЙ В ПРИЗРАЧНОМ ГРАФЕ:")
            if hasattr(prediction_manager, 'ghost_graph') and prediction_manager.ghost_graph:
                node_count = len(prediction_manager.ghost_graph.nodes)
                edge_count = len(prediction_manager.ghost_graph.edges)
                print(f"📊 Узлов в призрачном графе: {node_count}")
                print(f"📊 Связей в призрачном графе: {edge_count}")
                
                # Ожидаемое количество после объединения: 1 корень + 4 ребенка + 4 внука = 9
                expected_nodes = 9  
                if node_count == expected_nodes:
                    print(f"✅ Количество узлов корректно: {node_count} (ожидалось {expected_nodes})")
                else:
                    print(f"⚠️ Неожиданное количество узлов: {node_count} (ожидалось {expected_nodes})")
                    print(f"   Возможные дублирования: {node_count - expected_nodes}")
            else:
                print("❌ Призрачный граф недоступен для проверки")
                
            # Показываем статистику ПОСЛЕ операции  
            print("\n📊 СТАТИСТИКА ПОСЛЕ МЕРДЖА:")
            if hasattr(prediction_manager, 'get_ghost_graph_stats'):
                prediction_manager.get_ghost_graph_stats()
            if self.spore_manager and hasattr(self.spore_manager, 'get_graph_stats'):
                self.spore_manager.get_graph_stats()
                
            # Автоматически сохраняем debug картинку
            if hasattr(tree_logic, 'debug_plot_tree'):
                save_path = tree_logic.debug_plot_tree()
                print(f"💾 Debug картинка сохранена: {save_path}")
            else:
                print("⚠️ Метод debug_plot_tree не найден")
                
            print("="*60)
            print("✅ ОПЕРАЦИЯ МЕРДЖА ЗАВЕРШЕНА")
            print("="*60)
                
        except Exception as e:
            print(f"❌ Ошибка операции мерджа: {e}")
            import traceback
            traceback.print_exc()

    def _handle_all_graph_stats(self):
        """Обработчик статистики всех графов связей (L)."""
        try:
            print("\n" + "="*60)
            print("📊 ПОЛНАЯ СТАТИСТИКА ГРАФОВ СВЯЗЕЙ")
            print("="*60)
            
            # Реальный граф
            if self.spore_manager and hasattr(self.spore_manager, 'graph'):
                self.spore_manager.get_graph_stats()
            else:
                print("❌ Реальный граф не найден")
                
            # Призрачный граф
            if (self.manual_spore_manager and 
                hasattr(self.manual_spore_manager, 'prediction_manager') and
                hasattr(self.manual_spore_manager.prediction_manager, 'get_ghost_graph_stats')):
                self.manual_spore_manager.prediction_manager.get_ghost_graph_stats()
            else:
                print("❌ Призрачный граф не найден")
                
            # Сравнение с менеджерами
            print(f"\n📋 СРАВНЕНИЕ С МЕНЕДЖЕРАМИ:")
            if self.spore_manager:
                print(f"   🔸 Спор в SporeManager: {len(self.spore_manager.objects)}")
                print(f"   🔗 Линков в SporeManager: {len(self.spore_manager.links)}")
            if (self.manual_spore_manager and 
                hasattr(self.manual_spore_manager, 'prediction_manager')):
                pm = self.manual_spore_manager.prediction_manager
                if hasattr(pm, 'prediction_links'):
                    print(f"   👻 Призрачных линков: {len(pm.prediction_links)}")
                    
            print("="*60)
            
        except Exception as e:
            print(f"❌ Ошибка получения статистики графов: {e}")
            import traceback
            traceback.print_exc()

    def _handle_clear_all_graphs(self):
        """Обработчик очистки всех графов (Shift+L)."""
        try:
            if self.manual_spore_manager:
                print("\n🧹 ОЧИСТКА ВСЕХ ГРАФОВ СВЯЗЕЙ")
                self.manual_spore_manager.clear_all_graphs()
            else:
                print("❌ ManualSporeManager не найден")
        except Exception as e:
            print(f"❌ Ошибка очистки графов: {e}")

    def _handle_id_diagnostics(self):
        """Обработчик диагностики системы spore_id/link_id (Ctrl+I)."""
        if self.spore_manager:
            self.spore_manager.print_id_diagnostics()
        else:
            print("❌ SporeManager не найден")

    def _handle_graph_consistency(self):
        """Обработчик проверки консистентности граф vs объекты (Ctrl+Shift+I)."""
        if self.spore_manager:
            self.spore_manager.check_graph_id_consistency()
        else:
            print("❌ SporeManager не найден")

    def set_controls_window(self, controls_window: 'ControlsWindow'):
        """
        Связывает окно управления с InputManager.
        
        Args:
            controls_window: Экземпляр ControlsWindow для отображения команд
        """
        self.controls_window = controls_window
        print("📋 Controls window linked to InputManager")

    def _handle_toggle_controls(self):
        """Обработчик переключения видимости окна управления."""
        if self.controls_window:
            self.controls_window.toggle_visibility()
        else:
            print("⚠️ Controls window not initialized")

    def _handle_valence_analysis(self):
        """Обработчик анализа валентности (V)."""
        if not self.spore_manager:
            print("❌ SporeManager не найден")
            return

        try:
            # Импортируем ValenceManager
            from ..managers.valence_manager import ValenceManager

            # Создаем менеджер валентности
            valence_manager = ValenceManager(spore_manager=self.spore_manager)

            # Пытаемся получить спору под курсором через PickerManager
            target_spore_id = None

            if self.picker_manager:
                closest_spore = self.picker_manager.get_closest_spore()
                if closest_spore:
                    target_spore_id = closest_spore['id']
                    print(f"🎯 Анализ споры под курсором: {target_spore_id}")

            # Если PickerManager недоступен или не нашел спору, используем первую спору в графе
            if not target_spore_id:
                all_spore_ids = list(self.spore_manager.graph.nodes.keys())

                if not all_spore_ids:
                    print("❌ В графе нет спор для анализа")
                    return

                target_spore_id = all_spore_ids[0]
                print(f"ℹ️ PickerManager недоступен, анализ первой споры в графе: {target_spore_id}")

            print(f"\n🔬 АНАЛИЗ ВАЛЕНТНОСТИ СПОРЫ {target_spore_id}")
            print("=" * 60)

            # Выводим детальный отчет для выбранной споры
            valence_manager.print_valence_report(target_spore_id)

            # Выводим сводку по всему графу
            print("\n" + "=" * 60)
            valence_manager.print_graph_valence_summary()
            print("=" * 60)

        except Exception as e:
            print(f"❌ Ошибка анализа валентности: {e}")
            import traceback
            traceback.print_exc()
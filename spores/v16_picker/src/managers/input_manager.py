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

# Forward declaration для ManualSporeManager
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..managers.manual_spore_manager import ManualSporeManager
    from ..managers.angel_manager import AngelManager
    from ..visual.cost_visualizer import CostVisualizer
    from ..managers.dt_manager import DTManager
    
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
                 dt_manager: Optional['DTManager'] = None):
        
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

    def _setup_command_system(self):
        """Настраивает централизованную командную систему."""
        self.commands = {
            # === СПОРЫ ===
            'f': {
                'description': 'новая спора от последней (эволюция)',
                'handler': self._handle_spore_creation,
                'category': 'споры',
                'enabled': lambda: self.spore_manager is not None
            },
            'g': {
                'description': 'активировать случайную кандидатскую спору', 
                'handler': self._handle_candidate_activation,
                'category': 'споры',
                'enabled': lambda: self.spore_manager is not None
            },
            'v': {
                'description': 'развить всех кандидатов до завершения',
                'handler': self._handle_evolve_all,
                'category': 'споры', 
                'enabled': lambda: self.spore_manager is not None
            },
            
            # === ZOOM ===
            'e': {
                'description': 'приблизить камеру',
                'handler': self._handle_zoom_in,
                'category': 'зум',
                'enabled': lambda: self.zoom_manager is not None
            },
            't': {
                'description': 'отдалить камеру',
                'handler': self._handle_zoom_out,
                'category': 'зум',
                'enabled': lambda: self.zoom_manager is not None
            },
            'r': {
                'description': 'сброс всех трансформаций зума',
                'handler': self._handle_zoom_reset,
                'category': 'зум',
                'enabled': lambda: self.zoom_manager is not None
            },
            '1': {
                'description': 'уменьшить масштаб спор',
                'handler': self._handle_spores_scale_down,
                'category': 'зум',
                'enabled': lambda: self.zoom_manager is not None
            },
            '2': {
                'description': 'увеличить масштаб спор',
                'handler': self._handle_spores_scale_up,
                'category': 'зум',
                'enabled': lambda: self.zoom_manager is not None
            },
            
            # === КАНДИДАТЫ ===
            '5': {
                'description': 'уменьшить радиус генерации кандидатов',
                'handler': self._handle_candidates_radius_down,
                'category': 'кандидаты',
                'enabled': lambda: self.spore_manager is not None
            },
            '6': {
                'description': 'увеличить радиус генерации кандидатов',
                'handler': self._handle_candidates_radius_up,
                'category': 'кандидаты',
                'enabled': lambda: self.spore_manager is not None
            },
            
            # === ВИЗУАЛИЗАЦИЯ ===
            'y': {
                'description': 'включить/выключить ангелов',
                'handler': self._handle_toggle_angels,
                'category': 'визуализация',
                'enabled': lambda: self.angel_manager is not None
            },
            'u': {
                'description': 'переключить видимость системы координат',
                'handler': self._handle_toggle_frame,
                'category': 'визуализация',
                'enabled': lambda: self.scene_setup is not None
            },
            
            # === DT & ВРЕМЯ ===
            'm': {
                'description': 'сбросить dt к исходному значению',
                'handler': self._handle_dt_reset,
                'category': 'время',
                'enabled': lambda: self.dt_manager is not None
            },
            'j': {
                'description': 'показать статистику dt',
                'handler': self._handle_dt_stats,
                'category': 'время',
                'enabled': lambda: self.dt_manager is not None
            },
            
            # === ОПТИМИЗАЦИЯ ===
            'p': {
                'description': 'применить оптимальные пары к призрачному дереву',
                'handler': self._handle_optimal_pairs,
                'category': 'оптимизация',
                'enabled': lambda: self.manual_spore_manager is not None
            },
            
            # === ДЕРЕВЬЯ ===  
            'k': {
                'description': 'переключить режим создания (споры/деревья)',
                'handler': self._handle_toggle_creation_mode,
                'category': 'деревья',
                'enabled': lambda: self.manual_spore_manager is not None
            },
            '7': {
                'description': 'глубина дерева = 1',
                'handler': self._handle_tree_depth_1,
                'category': 'деревья',
                'enabled': lambda: self._is_tree_mode()
            },
            '8': {
                'description': 'глубина дерева = 2',
                'handler': self._handle_tree_depth_2,
                'category': 'деревья',
                'enabled': lambda: self._is_tree_mode()
            },
            
            # === ПРИЗРАКИ ===
            ';': {
                'description': 'включить/выключить призрачную систему', 
                'handler': self._handle_toggle_ghosts,
                'category': 'призраки',
                'enabled': lambda: self.manual_spore_manager is not None
            },
            
            # === ОТЛАДКА ===
            'h': {
                'description': 'переключить детальную отладку призрачного дерева',
                'handler': self._handle_debug_toggle,
                'category': 'отладка',
                'enabled': lambda: True
            },
            
            # === СПРАВКА ===
            'n': {
                'description': 'показать справку по всем командам',
                'handler': self._handle_help,
                'category': 'справка',
                'enabled': lambda: True
            },
            
            # === ДВИЖЕНИЕ КАМЕРЫ ===
            'w': {
                'description': 'движение камеры вперед',
                'handler': self._handle_move_forward,
                'category': 'движение',
                'enabled': lambda: self.scene_setup is not None
            },
            's': {
                'description': 'движение камеры назад',
                'handler': self._handle_move_backward,
                'category': 'движение',
                'enabled': lambda: self.scene_setup is not None
            },
            'a': {
                'description': 'движение камеры влево',
                'handler': self._handle_move_left,
                'category': 'движение',
                'enabled': lambda: self.scene_setup is not None
            },
            'd': {
                'description': 'движение камеры вправо',
                'handler': self._handle_move_right,
                'category': 'движение',
                'enabled': lambda: self.scene_setup is not None
            },
            'space': {
                'description': 'движение камеры вверх',
                'handler': self._handle_move_up,
                'category': 'движение',
                'enabled': lambda: self.scene_setup is not None
            },
            'shift': {
                'description': 'движение камеры вниз',
                'handler': self._handle_move_down,
                'category': 'движение',
                'enabled': lambda: self.scene_setup is not None
            },
            
                                            # === КУРСОР ===
                'alt': {
                    'description': 'переключить захват курсора (обрабатывается в SceneSetup)',
                    'handler': self._handle_toggle_cursor,
                    'category': 'курсор',
                    'enabled': lambda: self.scene_setup is not None
                },
            
            # === КОЛЕСИКО МЫШИ ===
            'scroll up': {
                'description': 'приблизить камеру (колесико вверх)',
                'handler': self._handle_scroll_up,
                'category': 'зум',
                'enabled': lambda: self.zoom_manager is not None
            },
            'scroll down': {
                'description': 'отдалить камеру (колесико вниз)',
                'handler': self._handle_scroll_down,
                'category': 'зум',
                'enabled': lambda: self.zoom_manager is not None
            }
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
        # v13_manual: обработка кликов мыши для создания спор
        if self.manual_spore_manager:
            current_mouse_left = mouse.left
            # Обнаруживаем нажатие ЛКМ (переход с False на True)
            if current_mouse_left and not self.previous_mouse_left:
                created_spores = self.manual_spore_manager.create_spore_at_cursor()
                if created_spores:
                    print(f"   🖱️ ЛКМ: Создано {len(created_spores)} спор (1 родитель + 2 ребёнка + 2 линка)")
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

    def _handle_help(self):
        """Обработчик вывода справки (N)."""
        self.print_commands_help()

    def _handle_move_forward(self):
        """Обработчик движения вперед (W)."""
        if self.scene_setup and self.scene_setup.player:
            from ursina import held_keys
            if held_keys['w']:  # type: ignore
                self.scene_setup.player.y += self.scene_setup.base_speed * 0.016  # Примерно 60 FPS

    def _handle_move_backward(self):
        """Обработчик движения назад (S)."""
        if self.scene_setup and self.scene_setup.player:
            from ursina import held_keys
            if held_keys['s']:  # type: ignore
                self.scene_setup.player.y -= self.scene_setup.base_speed * 0.016

    def _handle_move_left(self):
        """Обработчик движения влево (A)."""
        if self.scene_setup and self.scene_setup.player:
            from ursina import held_keys
            if held_keys['a']:  # type: ignore
                self.scene_setup.player.x -= self.scene_setup.base_speed * 0.016

    def _handle_move_right(self):
        """Обработчик движения вправо (D)."""
        if self.scene_setup and self.scene_setup.player:
            from ursina import held_keys
            if held_keys['d']:  # type: ignore
                self.scene_setup.player.x += self.scene_setup.base_speed * 0.016

    def _handle_move_up(self):
        """Обработчик движения вверх (Space)."""
        if self.scene_setup and self.scene_setup.player:
            from ursina import held_keys
            if held_keys['space']:  # type: ignore
                self.scene_setup.player.y += self.scene_setup.base_speed * 0.016

    def _handle_move_down(self):
        """Обработчик движения вниз (Shift)."""
        if self.scene_setup and self.scene_setup.player:
            from ursina import held_keys
            if held_keys['shift']:  # type: ignore
                self.scene_setup.player.y -= self.scene_setup.base_speed * 0.016

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

    def handle_input(self, key: str) -> None:
        """
        Новый централизованный обработчик команд через командную систему.
        """
        # Фильтруем события типа 'left alt up', 'right shift down', 'control' и т.д.
        if (' ' in key and any(direction in key.lower() for direction in ['up', 'down', 'left', 'right'])) or key == 'control':
            return  # Игнорируем события нажатия/отпускания модификаторов
        
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
                print(f"⚠️ Команда '{key}' недоступна (отключен {cmd_info['category']} менеджер)")
            return
        
        # Обработка специальной команды Ctrl+C через held_keys (совместимость)
        elif held_keys['c'] and held_keys['left control'] and self.spore_manager:  # type: ignore
            # Вызываем обработчик из словаря
            if 'ctrl+c' in self.commands:
                self.commands['ctrl+c']['handler']()
            return
        
        # Свободные клавиши
        elif key in ['z', 'x', 'c', 'i']:
            print(f"🔓 Клавиша '{key}' свободна")
            return
            
        # Неизвестная команда
        else:
            print(f"❓ Неизвестная команда: '{key}'. Нажмите 'N' для справки")

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
                show_debug=pairing_config.get('show_debug', True)
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
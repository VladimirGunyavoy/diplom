"""
UI Setup - Готовые настройки интерфейса для демо
================================================

Класс UI_setup инкапсулирует всю логику создания и управления UI
для демонстрационных скриптов. Избавляет от дублирования кода
и обеспечивает консистентный интерфейс во всех демо.

Особенности:
✅ Готовые настройки UI для разных сценариев
✅ Автоматическое создание всех необходимых элементов
✅ Интеграция с UI Manager и Color Manager
✅ Обработка команд и обновлений через колбэки
✅ Статистика и метрики в реальном времени
"""

import time
import numpy as np
from ursina import Entity, camera, color, Text
from .ui_manager import UIManager
from ..managers.color_manager import ColorManager
from textwrap import dedent
from .ui_constants import UI_POSITIONS
from ..logic.cost_function import CostFunction
from typing import Dict, Optional, Callable, Any
from ..logic.spawn_area import SpawnArea as SpawnAreaLogic


class UI_setup:
    """Готовые настройки UI для демонстрационных скриптов"""
    
    def __init__(self, 
                 color_manager: Optional[ColorManager] = None, 
                 ui_manager: Optional[UIManager] = None):
        # Менеджеры
        self.color_manager: ColorManager = color_manager if color_manager is not None else ColorManager()
        self.ui_manager: UIManager = ui_manager if ui_manager is not None else UIManager(self.color_manager)

        self.cost_function: CostFunction = CostFunction(goal_position_2d=np.array([np.pi, 0]))
        
        # Состояние
        self.start_time: float = time.time()
        self.ui_elements: Dict[str, Entity] = {}
        self.ui_toggle_timer: float = 0  # Для обработки клавиши H
        
        # Хранилище функций-поставщиков данных
        self.data_providers: Dict[str, Callable[[], Any]] = {}
    
    def setup_spawn_area_ui(self, spawn_area: SpawnAreaLogic) -> None:
        """Создает UI для SpawnArea и настраивает колбэки."""
        print("   ✓ UI для SpawnArea")

        # Создаем текст для эксцентриситета
        self.ui_elements['eccentricity_info'] = self.ui_manager.create_element(
            'dynamic', 'eccentricity_info',
            text=f'ECCENTRICITY: {spawn_area.eccentricity:.3f}',
            position=UI_POSITIONS.SPAWN_AREA_INFO,
            style='header' 
        )

        # Создаем инструкции
        instructions_text = dedent('''
            <white>SPAWN AREA:
            3 - increase eccentricity
            4 - decrease eccentricity
        ''').strip()
        self.ui_elements['spawn_area_instructions'] = self.ui_manager.create_instructions(
            'spawn_area',
            instructions_text,
            position=UI_POSITIONS.SPAWN_AREA_INSTRUCTIONS,
        )

        # Функция для обновления текста
        def update_eccentricity_text(new_eccentricity: float) -> None:
            self.ui_manager.update_text('eccentricity_info', f'ECCENTRICITY: {new_eccentricity:.3f}')

        # Регистрируем колбэк
        if hasattr(spawn_area, 'on_eccentricity_change_callbacks'):
            spawn_area.on_eccentricity_change_callbacks.append(update_eccentricity_text)

    def setup_demo_ui(self, 
                      data_providers: Optional[Dict[str, Callable[[], Any]]] = None, 
                      spawn_area: Optional[SpawnAreaLogic] = None) -> Dict[str, Entity]:
        """
        Настраивает полный UI для демонстрационного скрипта, используя колбэки.
        
        Args:
            data_providers (dict): Словарь с функциями для получения данных.
                                   Пример: {'get_spore_count': lambda: ...}
        """
        # Сохраняем ссылки на поставщиков данных
        self.data_providers = data_providers if data_providers else {}
        
        print("\n🎨 UI_setup: Настройка демонстрационного интерфейса...")
        
        # 1. Основной UI сцены (позиция камеры и статус курсора)
        if 'get_camera_info' in self.data_providers and 'get_cursor_status' in self.data_providers:
            self._create_scene_ui()
            print("   ✓ Основной UI сцены")
        
        # 2. Zoom UI (если есть провайдер) 
        if 'get_look_point_info' in self.data_providers:
            self._create_zoom_ui()
            print("   ✓ Zoom UI")
        
        # 3. Param Manager UI (если есть провайдер)
        if 'get_param_info' in self.data_providers:
            self._create_param_ui()
            print("   ✓ Param Manager UI")
        
        # 4. Счетчик спор и команда создания
        if 'get_spore_count' in self.data_providers:
            self.ui_elements['spore_counter'] = self.ui_manager.create_counter(
                'spore_count', 0,
                position=UI_POSITIONS.SPORE_COUNTER,
                prefix="Spores: "
            )
            
            # Добавляем инструкцию "F - new spore" через UI Manager
            self.ui_elements['spore_instruction'] = self.ui_manager.create_instructions(
                'spore_control',
                'F - new spore',
                position=UI_POSITIONS.SPORE_INSTRUCTION
            )
            print("   ✓ Счетчик спор и команда F")
        
        # 4.5. Информация о кандидатах
        if 'get_candidate_info' in self.data_providers:
            min_radius, candidate_count = self.data_providers['get_candidate_info']()
            self.ui_elements['candidate_info'] = self.ui_manager.create_element(
                'dynamic', 'candidate_info',
                text=f'Candidates: {candidate_count}\nMin radius: {min_radius:.2f}',
                position=UI_POSITIONS.CANDIDATE_INFO,
                style='normal'
            )
            
            # Добавляем инструкции по управлению
            self.ui_elements['candidate_controls'] = self.ui_manager.create_instructions(
                'candidate_controls',
                'G - activate candidate\n5/6 - adjust radius',
                position=UI_POSITIONS.CANDIDATE_CONTROLS
            )
            print("   ✓ Информация о кандидатах")
        
        # X. UI для SpawnArea
        if spawn_area:
            self.setup_spawn_area_ui(spawn_area)
        
        # 5. Статистика UI убрана
        print("   ✓ Блок статистики убран")
        
        # 6. Отладочная информация убрана
        print("   ✓ Отладочная информация отключена")
        
        # 7. Игровые команды
        self._create_game_commands()
        print("   ✓ Игровые команды")
        
        # 8. Демонстрационные команды убраны
        print("   ✓ Демонстрационные команды убраны")
        
        # 9. Регистрируем функции обновления
        self._register_update_functions()
        print("   ✓ Функции автообновления")
        
        print(f"   📊 Всего UI элементов: {self.ui_manager.get_stats()['total']}")
        return self.ui_elements
    
    def _create_game_commands(self) -> None:
        """Создает блок с основными игровыми командами"""
        game_text = """CONTROLS:
WASD - move camera
Space/Shift - up/down
E/T - zoom in/out
R - reset zoom
1 - larger spores
2 - smaller spores
H - hide/show all UI"""
        
        self.ui_elements['game_commands'] = self.ui_manager.create_instructions(
            'game_controls',
            game_text,
            position=UI_POSITIONS.GAME_CONTROLS
        )
    
    def _create_scene_ui(self) -> None:
        """Создает основные UI элементы сцены"""
        # Позиционная информация
        self.ui_elements['position_info'] = self.ui_manager.create_position_info()
        
        # Статус курсора  
        self.ui_elements['cursor_status'] = self.ui_manager.create_status_indicator(
            'cursor', "Cursor locked [Alt to unlock]"
        )
    
    def _create_zoom_ui(self) -> None:
        """Создает UI элементы для zoom manager"""
        # Информация о точке взгляда
        self.ui_elements['look_point'] = self.ui_manager.create_debug_info(
            'look_point', position=UI_POSITIONS.LOOK_POINT_DEBUG
        )
    
    def _create_param_ui(self) -> None:
        """Создает UI элементы для param manager"""
        param_value, show_param = self.data_providers['get_param_info']()
        
        self.ui_elements['param_value'] = self.ui_manager.create_element(
            'dynamic', 'param_value',  # Явно указываем категорию
            text=f'param value: {round(param_value, 3)}',
            position=UI_POSITIONS.PARAM_VALUE,
            style='header'
        )
        
        # Скрываем если show=False
        if not show_param:
            self.ui_manager.hide_element('param_value')
    
    # Метод _create_demo_commands() удален - плашка UI CONTROL больше не нужна
    
    def _register_update_functions(self) -> None:
        """Регистрирует все функции автообновления"""
        
        # Обновление счетчика спор
        def update_counters() -> None:
            # Счетчик спор
            if 'get_spore_count' in self.data_providers:
                spore_count = self.data_providers['get_spore_count']()
                self.ui_manager.update_counter('spore_count', spore_count, prefix="Spores: ")
            
        self.ui_manager.register_update_function('counters', update_counters)
        
        # Обновление позиции камеры
        def update_position() -> None:
            if 'get_camera_info' in self.data_providers:
                try:
                    pos_x, pos_y, pos_z, rot_x, rot_y, rot_z = self.data_providers['get_camera_info']()
                    text = f"Position: {pos_x:>7.3f}, {pos_y:>7.3f}, {pos_z:>7.3f}"
                    text += f"\nRotation: {rot_x:>7.3f}, {rot_y:>7.3f}, {rot_z:>7.3f}"
                    self.ui_manager.update_text('main', text)
                except Exception:
                    pass
        
        self.ui_manager.register_update_function('position_info', update_position)
        
        # Обновление статуса курсора
        def update_cursor_status() -> None:
            if 'get_cursor_status' in self.data_providers:
                is_locked = self.data_providers['get_cursor_status']()
                if is_locked:
                    self.ui_manager.update_text('cursor', "Cursor locked [Alt to unlock]")
                else:
                    self.ui_manager.update_text('cursor', "Cursor unlocked [Alt to lock]")
        
        self.ui_manager.register_update_function('cursor_status', update_cursor_status)
        
        # Обновление точки взгляда (для zoom manager)
        def update_look_point() -> None:
            if 'get_look_point_info' in self.data_providers:
                try:
                    x_0, z_0, current_scale, spores_scale = self.data_providers['get_look_point_info']()
                    text = f'LOOK POINT:\n({x_0:6.3f}, {z_0:6.3f})\n'
                    text += f'cost: {np.round(self.cost_function.get_cost(np.array([x_0, z_0])), 3)}\n'
                    text += f'common scale: {current_scale:6.3f}\n'
                    text += f'spores scale: {spores_scale:6.3f}'
                    self.ui_manager.update_text('look_point', text)
                except Exception:
                    pass
        
        self.ui_manager.register_update_function('look_point', update_look_point)

        # Обновление параметра (для param manager)
        def update_param_value() -> None:
            if 'get_param_info' in self.data_providers:
                try:
                    param_value, show_param = self.data_providers['get_param_info']()
                    # Обновляем видимость и текст
                    element = self.ui_manager.get_element('param_value')
                    if element:
                        element.enabled = show_param
                        if show_param:
                            self.ui_manager.update_text('param_value', f'param value: {round(param_value, 3)}')
                except Exception:
                    pass # Игнорируем ошибки, если данные еще не готовы
        
        self.ui_manager.register_update_function('param_value', update_param_value)
        
        # Обновление информации о кандидатах
        def update_candidate_info() -> None:
            if 'get_candidate_info' in self.data_providers:
                try:
                    min_radius, candidate_count = self.data_providers['get_candidate_info']()
                    text = f'Candidates: {candidate_count}\nMin radius: {min_radius:.2f}'
                    self.ui_manager.update_text('candidate_info', text)
                except Exception:
                    pass
        
        self.ui_manager.register_update_function('candidate_info', update_candidate_info)


    # ===== ОБРАБОТЧИК КОМАНД =====
    def handle_demo_commands(self, key: str) -> bool:
        """Обрабатывает команды, специфичные для этого UI."""
        if key == 'h' and time.time() - self.ui_toggle_timer > 0.2:
            self.ui_manager.toggle_category('static')
            self.ui_manager.toggle_category('dynamic')
            self.ui_toggle_timer = time.time()
            return True
        return False
        
    def update(self) -> None:
        """Вызывает все зарегистрированные функции обновления UI."""
        self.ui_manager.update_dynamic_elements()

    def setup_architecture_demo(self) -> None:
        """Настраивает UI для демонстрации архитектуры."""
        self.ui_manager.create_element('static', 'sim_panel', style='header', position=UI_POSITIONS.SIMULATION_PANEL, text="SIMULATION")
        self.ui_manager.create_element('static', 'opt_panel', style='header', position=UI_POSITIONS.OPTIMIZATION_PANEL, text="OPTIMIZATION")
        
        # Счетчик операций
        self.ui_manager.create_counter('op_counter', 0, position=UI_POSITIONS.OPERATION_COUNTER, prefix="Operations: ")

    def update_simulation_stats(self, spore_count: int, steps_per_sec: float) -> None:
        """Обновляет статистику симуляции."""
        self.ui_manager.update_counter('spore_count', spore_count, prefix="Spores: ")
        # Добавьте другие счетчики, если нужно...

    def update_optimization_stats(self, iterations: int, nfev: int, cost: float) -> None:
        """Обновляет статистику оптимизации."""
        # Здесь можно обновлять UI элементы, связанные с оптимизацией.
        pass

    def update_operation_counter(self, count: int) -> None:
        """Обновляет счетчик операций."""
        self.ui_manager.update_counter('op_counter', count, prefix="Operations: ")

    def create_info_panel(self, title: str, content: str, position: tuple = (-0.95, 0.5)) -> Entity:
        """Создает информационную панель."""
        return self.ui_manager.create_info_block('info_panel', title, content, position)

    def create_status_panel(self, name: str, text: str, position: tuple = (0, -0.3)) -> Entity:
        """Создает статусную панель."""
        return self.ui_manager.create_status_indicator(name, text, position)
    
    def show_demo_commands_help(self) -> None:
        """Показывает справку по командам демо."""
        help_text = dedent('''
            <white>UI CONTROL:
            H - toggle all UI
            L - show layout map
        ''').strip()

        self.ui_manager.create_instructions(
            'demo_help',
            help_text,
            position=UI_POSITIONS.DEMO_COMMANDS,
        )

    def show_ui_layout_map(self) -> None:
        """Визуализирует все константы из UI_POSITIONS для отладки."""
        # Отключаем основной UI, чтобы не мешал
        self.ui_manager.hide_category('static')
        self.ui_manager.hide_category('dynamic')

        # Создаем временные метки
        for name, pos in UI_POSITIONS.__dict__.items():
            if not name.startswith('__') and isinstance(pos, tuple):
                Text(text=name, position=pos, scale=0.6, color=color.orange, origin=(0,0))

        # Создаем инструкцию по возврату
        Text(text="Press 'L' again to hide layout map", 
             position=(-0.1, -0.45), 
             color=color.white,
             scale=0.8)

        def input_handler(key):
            if key == 'l':
                # Уничтожаем все временные метки и показываем основной UI
                for entity in camera.ui.children:
                    if isinstance(entity, Text) and entity.color == color.orange:
                        destroy(entity)
                destroy(self.input_handler) # Удаляем сам обработчик
                self.ui_manager.show_category('static')
                self.ui_manager.show_category('dynamic')

        self.input_handler = input_handler
        
    def get_stats(self) -> Dict[str, int]:
        """Возвращает статистику по UI элементам."""
        return self.ui_manager.get_stats()

# Для обратной совместимости, если где-то используется старый вызов
def create_demo_ui(
    color_manager: Optional[ColorManager] = None, 
    ui_manager: Optional[UIManager] = None
) -> UI_setup:
    return UI_setup(color_manager, ui_manager)
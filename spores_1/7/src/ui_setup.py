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
from .ui_manager import UIManager
from .color_manager import ColorManager
from textwrap import dedent


class UI_setup:
    """Готовые настройки UI для демонстрационных скриптов"""
    
    def __init__(self, color_manager=None, ui_manager=None):
        # Менеджеры
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager
        
        if ui_manager is None:
            ui_manager = UIManager(color_manager)
        self.ui_manager = ui_manager
        
        # Состояние
        self.start_time = time.time()
        self.ui_elements = {}
        self.ui_toggle_timer = 0  # Для обработки клавиши H
        
        # Хранилище функций-поставщиков данных
        self.data_providers = {}
    
    def setup_spawn_area_ui(self, spawn_area):
        """Создает UI для SpawnArea и настраивает колбэки."""
        print("   ✓ UI для SpawnArea")

        # Создаем текст для эксцентриситета
        self.ui_elements['eccentricity_info'] = self.ui_manager.create_element(
            'dynamic', 'eccentricity_info',
            text=f'ECCENTRICITY: {spawn_area.eccentricity:.3f}',
            position=(0.55, 0.3),
            style='header' 
        )

        # Создаем инструкции
        instructions_text = dedent('''
            <white>SPAWN AREA:
            3 - decrease eccentricity
            4 - increase eccentricity
        ''').strip()
        self.ui_elements['spawn_area_instructions'] = self.ui_manager.create_instructions(
            'spawn_area',
            instructions_text,
            position=(0.5, 0.21),
        )

        # Функция для обновления текста
        def update_eccentricity_text(new_eccentricity):
            self.ui_manager.update_text('eccentricity_info', f'ECCENTRICITY: {new_eccentricity:.3f}')

        # Регистрируем колбэк
        spawn_area.on_eccentricity_change_callbacks.append(update_eccentricity_text)

    def setup_demo_ui(self, data_providers=None, spawn_area=None):
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
                position=(-0.76, 0.2),
                prefix="Spores: "
            )
            
            # Добавляем инструкцию "F - new spore" через UI Manager
            self.ui_elements['spore_instruction'] = self.ui_manager.create_instructions(
                'spore_control',
                'F - new spore',
                position=(-0.76, 0.27) # Пониже, чтобы не мешать
            )
            print("   ✓ Счетчик спор и команда F")
        
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
    
    def _create_game_commands(self):
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
            position=(-0.76, 0.46)  # Возвращаем в видимую область
        )
    
    def _create_scene_ui(self):
        """Создает основные UI элементы сцены"""
        # Позиционная информация
        self.ui_elements['position_info'] = self.ui_manager.create_position_info()
        
        # Статус курсора  
        self.ui_elements['cursor_status'] = self.ui_manager.create_status_indicator(
            'cursor', "Cursor locked [Alt to unlock]"
        )
    
    def _create_zoom_ui(self):
        """Создает UI элементы для zoom manager"""
        # Информация о точке взгляда
        self.ui_elements['look_point'] = self.ui_manager.create_debug_info(
            'look_point', position=(0.5, 0.47)
        )
    
    def _create_param_ui(self):
        """Создает UI элементы для param manager"""
        param_value, show_param = self.data_providers['get_param_info']()
        
        self.ui_elements['param_value'] = self.ui_manager.create_element(
            'dynamic', 'param_value',  # Явно указываем категорию
            text=f'param value: {round(param_value, 3)}',
            position=(0.57, -0.31),
            style='header'
        )
        
        # Скрываем если show=False
        if not show_param:
            self.ui_manager.hide_element('param_value')
    
    # Метод _create_demo_commands() удален - плашка UI CONTROL больше не нужна
    
    def _register_update_functions(self):
        """Регистрирует все функции автообновления"""
        
        # Обновление счетчика спор
        def update_counters():
            # Счетчик спор
            if 'get_spore_count' in self.data_providers:
                spore_count = self.data_providers['get_spore_count']()
                self.ui_manager.update_counter('spore_count', spore_count, prefix="Spores: ")
            
        self.ui_manager.register_update_function('counters', update_counters)
        
        # Обновление позиции камеры
        def update_position():
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
        def update_cursor_status():
            if 'get_cursor_status' in self.data_providers:
                is_locked = self.data_providers['get_cursor_status']()
                if is_locked:
                    self.ui_manager.update_text('cursor', "Cursor locked [Alt to unlock]")
                else:
                    self.ui_manager.update_text('cursor', "Cursor unlocked [Alt to lock]")
        
        self.ui_manager.register_update_function('cursor_status', update_cursor_status)
        
        # Обновление точки взгляда (для zoom manager)
        def update_look_point():
            if 'get_look_point_info' in self.data_providers:
                try:
                    x_0, z_0, current_scale, spores_scale = self.data_providers['get_look_point_info']()
                    text = f'LOOK POINT:\n({x_0:6.3f}, {z_0:6.3f})\n'
                    text += f'common scale: {current_scale:6.3f}\n'
                    text += f'spores scale: {spores_scale:6.3f}'
                    self.ui_manager.update_text('look_point', text)
                except Exception:
                    pass
        
        self.ui_manager.register_update_function('look_point', update_look_point)

        # Обновление параметра (для param manager)
        def update_param_value():
            if 'get_param_info' in self.data_providers:
                try:
                    param_value, show_param = self.data_providers['get_param_info']()
                    element = self.ui_manager.get_element('param_value')
                    if element:
                        element.enabled = show_param
                        if show_param:
                            element.text = f'param value: {round(param_value, 3)}'
                except Exception:
                    pass
        
        self.ui_manager.register_update_function('param_value', update_param_value)
    
    def handle_demo_commands(self, key):
        """
        Обработка команд демонстрации UI
        
        Args:
            key: Нажатая клавиша
            
        Returns:
            bool: True если команда была обработана, False иначе
        """
        if key == 'h':
            # Глобальное выключение/включение всего UI
            if time.time() - self.ui_toggle_timer > 0.3:
                # Получаем все реальные категории UI
                all_categories = list(self.ui_manager.elements.keys())
                
                # Проверяем есть ли хотя бы один видимый элемент
                any_visible = False
                for category in all_categories:
                    elements = self.ui_manager.elements.get(category, {})
                    for element in elements.values():
                        if element.enabled:
                            any_visible = True
                            break
                    if any_visible:
                        break
                
                # Если есть видимые элементы - скрываем ВСЕ, иначе показываем ВСЕ
                if any_visible:
                    for category in all_categories:
                        self.ui_manager.hide_category(category)
                    print("🙈 Весь UI скрыт (H)")
                else:
                    for category in all_categories:
                        self.ui_manager.show_category(category)
                    print("👁️ Весь UI показан (H)")
                
                self.ui_toggle_timer = time.time()
            return True
        
        return False
    
    def update(self):
        """Обновляет все UI элементы (вызывается из основного цикла)"""
        self.ui_manager.update_dynamic_elements()
    
    def setup_architecture_demo(self):
        """Специальная настройка для демонстрации архитектуры (как в scripts/2.py)"""
        from textwrap import dedent
        
        # Информация об архитектуре
        architecture_text = dedent('''
        НОВАЯ АРХИТЕКТУРА SPORE
        🧮 SporeLogic: чистая математика (2D)
        🎨 SporeVisual: 3D визуализация
        🔄 Spore: логика + визуализация
        ✅ 100% совместимость с 1.py
        ''').strip()
        
        self.ui_elements['architecture_info'] = self.ui_manager.create_info_block(
            'architecture', architecture_text
        )
        
        # Счетчик операций
        self.ui_elements['operation_counter'] = self.ui_manager.create_counter(
            'operations', 0, prefix="Операций: "
        )
        
        # Инструкции тестирования
        test_instructions = dedent('''
        🔬 T - тест архитектуры
        🧮 L - тест чистой логики
        🎨 V - тест визуализации
        🔄 S - синхронизация
        📊 I - информация
        ''').strip()
        
        self.ui_elements['test_instructions'] = self.ui_manager.create_instructions(
            'test_controls', test_instructions, position=(-0.95, 0.1)
        )
        self.ui_elements['test_instructions'].scale = 0.6
        try:
            from ursina import color
            self.ui_elements['test_instructions'].color = color.cyan
        except:
            pass
        
        # Дополнительные элементы для архитектурной демонстрации
        operation_counter = self.ui_manager.create_counter(
            'operations', 0,
            position=(-0.95, 0.3),
            prefix="Операций: "
        )
        self.ui_elements['operation_counter'] = operation_counter
        
        return arch_ui
    
    def update_operation_counter(self, count):
        """Обновляет счетчик операций для архитектурной демонстрации"""
        self.ui_manager.update_counter('operations', count, prefix="Операций: ")
    
    def create_info_panel(self, title, content, position=(-0.95, 0.5)):
        """Создает информационную панель"""
        return self.ui_manager.create_info_block(
            f'panel_{title.lower()}', title, content, position
        )
    
    def create_status_panel(self, name, text, position=(0, -0.3)):
        """Создает статусную панель"""
        return self.ui_manager.create_status_indicator(name, text, position)
    
    def show_demo_commands_help(self):
        """Печатает справку по всем командам"""
        print("\n🎮 ИГРОВЫЕ КОМАНДЫ:")
        print("   F - создать новую спору")
        print("   WASD - движение камеры")
        print("   Space/Shift - движение вверх/вниз")
        print("   E/T - увеличить/уменьшить зум")
        print("   R - сбросить зум")
        print("   1/2 - размер спор")
        print("   Z/X - параметр (если включен)")
        
        print("\n🎨 UI УПРАВЛЕНИЕ:")
        print("   H - скрыть/показать весь UI")
    
    def show_ui_layout_map(self):
        """Показывает карту расположения всех UI элементов"""
        print("\n📍 КАРТА РАСПОЛОЖЕНИЯ UI ЭЛЕМЕНТОВ:")
        print("=" * 50)
        print("ЛЕВАЯ СТОРОНА (X < 0):")
        print("  (-0.76, 0.46)  GAME CONTROLS")
        print("  (-0.76, 0.0)   SPORE COUNTER")
        print("  (-0.76, -0.4)  F - NEW SPORE")
        print()
        print("ПРАВАЯ СТОРОНА (X > 0):")
        print("  (0.4, -0.4)    POSITION INFO")
        print("  (0.0, -0.3)    CURSOR STATUS")
        print("  (0.5, 0.47)    LOOK POINT DEBUG")
        print("  (0.57, -0.31)  PARAM VALUE")
        print()
        print("ЦЕНТР:")
        print("  (0, -0.4)      CURSOR STATUS")
        print("=" * 50)
    
    def get_stats(self):
        """Возвращает статистику UI"""
        elapsed = time.time() - self.start_time
        
        return {
            'ui_elements': self.ui_manager.get_stats(),
            'elapsed_time': elapsed,
            'systems': {
                'has_camera_provider': 'get_camera_info' in self.data_providers,
                'has_zoom_provider': 'get_look_point_info' in self.data_providers,
                'has_param_provider': 'get_param_info' in self.data_providers,
                'has_spore_provider': 'get_spore_count' in self.data_providers,
            }
        }


# Удобная функция для быстрого создания UI_setup
def create_demo_ui(color_manager=None, ui_manager=None):
    """Создает и возвращает настроенный UI_setup"""
    return UI_setup(color_manager, ui_manager) 
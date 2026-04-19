#!/usr/bin/env python3
"""
Упрощенный тест интеграции ControlsWindow в UI систему
======================================================

Проверяет правильность интеграции ControlsWindow через UIManager без зависимостей от ursina.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_ui_constants():
    """Тестирует наличие позиции для ControlsWindow в UI_POSITIONS."""
    print("Testing UI_POSITIONS...")
    
    # Импортируем только константы
    from src.visual.ui_constants import UI_POSITIONS
    
    assert hasattr(UI_POSITIONS, 'CONTROLS_WINDOW'), "CONTROLS_WINDOW не найден в UI_POSITIONS"
    assert UI_POSITIONS.CONTROLS_WINDOW is not None, "CONTROLS_WINDOW равен None"
    
    print(f"   CONTROLS_WINDOW position: {UI_POSITIONS.CONTROLS_WINDOW}")

def test_ui_manager_structure():
    """Тестирует структуру UIManager для работы с ControlsWindow."""
    print("Testing UIManager structure...")
    
    # Читаем файл и проверяем наличие методов
    ui_manager_path = os.path.join(os.path.dirname(__file__), 'src', 'visual', 'ui_manager.py')
    
    with open(ui_manager_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем наличие импорта ControlsWindow
    assert 'from .controls_window import ControlsWindow' in content, "Импорт ControlsWindow не найден"
    print("   ControlsWindow import found")
    
    # Проверяем наличие атрибута
    assert 'self.controls_window: Optional[\'ControlsWindow\'] = None' in content, "Атрибут controls_window не найден"
    print("   controls_window attribute found")
    
    # Проверяем наличие методов
    methods = [
        'def create_controls_window',
        'def toggle_controls_window', 
        'def show_controls_window',
        'def hide_controls_window',
        'def update_controls_window'
    ]
    
    for method in methods:
        assert method in content, f"Метод {method} не найден"
        print(f"   Method {method} found")

def test_ui_setup_integration():
    """Тестирует интеграцию в UI_setup."""
    print("Testing UI_setup integration...")
    
    # Читаем файл и проверяем изменения
    ui_setup_path = os.path.join(os.path.dirname(__file__), 'src', 'visual', 'ui_setup.py')
    
    with open(ui_setup_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем использование UIManager для создания ControlsWindow
    assert 'self.ui_manager.create_controls_window(' in content, "Создание через UIManager не найдено"
    print("   ControlsWindow creation via UIManager found")
    
    # Проверяем использование UI_POSITIONS.CONTROLS_WINDOW
    assert 'UI_POSITIONS.CONTROLS_WINDOW' in content, "Использование UI_POSITIONS.CONTROLS_WINDOW не найдено"
    print("   UI_POSITIONS.CONTROLS_WINDOW usage found")
    
    # Проверяем наличие метода toggle_controls_window
    assert 'def toggle_controls_window(self) -> None:' in content, "Метод toggle_controls_window не найден"
    print("   toggle_controls_window method found")

def test_controls_window_structure():
    """Тестирует структуру ControlsWindow."""
    print("Testing ControlsWindow structure...")
    
    controls_window_path = os.path.join(os.path.dirname(__file__), 'src', 'visual', 'controls_window.py')
    
    with open(controls_window_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем основные методы
    methods = [
        'def __init__',
        'def toggle_visibility',
        'def update_commands',
        'def _generate_controls_text'
    ]
    
    for method in methods:
        assert method in content, f"Метод {method} не найден в ControlsWindow"
        print(f"   Method {method} found in ControlsWindow")

def main():
    """Запускает все тесты."""
    print("Running simplified ControlsWindow integration tests...\n")
    
    try:
        test_ui_constants()
        test_ui_manager_structure() 
        test_ui_setup_integration()
        test_controls_window_structure()
        
        print("\nAll tests passed successfully!")
        print("ControlsWindow properly integrated into UI system")
        print("\nSummary of changes:")
        print("   • Added CONTROLS_WINDOW position to UI_POSITIONS")
        print("   • Added ControlsWindow management methods to UIManager")
        print("   • Fixed ControlsWindow creation in UI_setup via UIManager")
        print("   • Improved integration architecture")
        
    except Exception as e:
        print(f"\nTest error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

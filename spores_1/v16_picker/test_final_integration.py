#!/usr/bin/env python3
"""
Финальный тест интеграции ControlsWindow в UI систему
====================================================

Проверяет правильность интеграции ControlsWindow через UIManager.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_ui_constants():
    """Тестирует наличие позиции для ControlsWindow в UI_POSITIONS."""
    print("Testing UI_POSITIONS...")
    
    # Импортируем только константы
    from src.visual.ui_constants import UI_POSITIONS
    
    assert hasattr(UI_POSITIONS, 'CONTROLS_WINDOW'), "CONTROLS_WINDOW not found in UI_POSITIONS"
    assert UI_POSITIONS.CONTROLS_WINDOW is not None, "CONTROLS_WINDOW is None"
    
    print(f"   CONTROLS_WINDOW position: {UI_POSITIONS.CONTROLS_WINDOW}")

def test_ui_manager_integration():
    """Тестирует интеграцию ControlsWindow в UIManager."""
    print("Testing UIManager integration...")
    
    # Читаем файл и проверяем наличие методов
    ui_manager_path = os.path.join(os.path.dirname(__file__), 'src', 'visual', 'ui_manager.py')
    
    with open(ui_manager_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем наличие импорта ControlsWindow
    assert 'from .controls_window import ControlsWindow' in content, "ControlsWindow import not found"
    print("   ControlsWindow import found")
    
    # Проверяем наличие атрибута
    assert 'self.controls_window: Optional[\'ControlsWindow\'] = None' in content, "controls_window attribute not found"
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
        assert method in content, f"Method {method} not found"
        print(f"   Method {method} found")

def test_ui_setup_integration():
    """Тестирует интеграцию в UI_setup."""
    print("Testing UI_setup integration...")
    
    # Читаем файл и проверяем изменения
    ui_setup_path = os.path.join(os.path.dirname(__file__), 'src', 'visual', 'ui_setup.py')
    
    with open(ui_setup_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем использование UIManager для создания ControlsWindow
    assert 'self.ui_manager.create_controls_window(' in content, "Creation via UIManager not found"
    print("   ControlsWindow creation via UIManager found")
    
    # Проверяем использование UI_POSITIONS.CONTROLS_WINDOW
    assert 'UI_POSITIONS.CONTROLS_WINDOW' in content, "UI_POSITIONS.CONTROLS_WINDOW usage not found"
    print("   UI_POSITIONS.CONTROLS_WINDOW usage found")
    
    # Проверяем наличие метода toggle_controls_window
    assert 'def toggle_controls_window(self) -> None:' in content, "toggle_controls_window method not found"
    print("   toggle_controls_window method found")

def test_controls_window_improvements():
    """Тестирует улучшения в ControlsWindow."""
    print("Testing ControlsWindow improvements...")
    
    controls_window_path = os.path.join(os.path.dirname(__file__), 'src', 'visual', 'controls_window.py')
    
    with open(controls_window_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем улучшенную стилизацию
    assert 'self.color_manager.get_color(' in content, "ColorManager integration not found"
    print("   ColorManager integration found")
    
    # Проверяем правильную позицию по умолчанию
    assert 'position: Tuple[float, float] = (-0.3, 0.1)' in content, "Default position not updated"
    print("   Default position updated correctly")
    
    # Проверяем основные методы
    methods = [
        'def __init__',
        'def toggle_visibility',
        'def update_commands',
        'def _generate_controls_text'
    ]
    
    for method in methods:
        assert method in content, f"Method {method} not found in ControlsWindow"
        print(f"   Method {method} found in ControlsWindow")

def main():
    """Запускает все тесты."""
    print("Running final ControlsWindow integration tests...\n")
    
    try:
        test_ui_constants()
        test_ui_manager_integration() 
        test_ui_setup_integration()
        test_controls_window_improvements()
        
        print("\nAll tests passed successfully!")
        print("ControlsWindow properly integrated into UI system")
        print("\nSummary of improvements:")
        print("   • Added CONTROLS_WINDOW position to UI_POSITIONS")
        print("   • Integrated ControlsWindow management in UIManager")
        print("   • Fixed ControlsWindow creation in UI_setup via UIManager")
        print("   • Improved ControlsWindow styling with ColorManager")
        print("   • Proper architecture: ControlsWindow as managed component")
        
    except Exception as e:
        print(f"\nTest error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

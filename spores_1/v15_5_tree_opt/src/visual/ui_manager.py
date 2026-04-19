"""
UI Manager - Централизованное управление всеми UI элементами
============================================================

Класс для централизованного управления всеми текстовыми элементами в проекте:
- Статичные плашки (инструкции)
- Динамические плашки (статусы, счетчики)

Преимущества:
✅ Все UI элементы в одном месте
✅ Упрощенная структура (статичные/динамичные)
✅ Легкое обновление и управление
✅ Поддержка динамического обновления
"""

from ursina import color, destroy, Text
from typing import Dict, Optional, Callable, Any
from ..managers.color_manager import ColorManager
from .ui_constants import UI_POSITIONS


class UIManager:
    """Централизованный менеджер UI элементов"""
    
    def __init__(self, color_manager: Optional[ColorManager] = None):
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager: ColorManager = color_manager
        
        self.elements: Dict[str, Dict[str, Text]] = {'static': {}, 'dynamic': {}}
        self.update_functions: Dict[str, Callable[[], None]] = {}
        
        self.styles: Dict[str, Dict[str, Any]] = {
            'default': {
                'scale': 0.7, 'color': self.color_manager.get_color('ui', 'text_primary'),
                'font': 'VeraMono.ttf', 'has_background': True,
            },
            'instructions': {
                'scale': 0.7, 'color': self.color_manager.get_color('ui', 'text_primary'),
                'font': 'VeraMono.ttf', 'has_background': True,
            },
            'header': {
                'scale': 0.7, 'color': self.color_manager.get_color('ui', 'text_primary'),
                'font': 'VeraMono.ttf', 'has_background': True,
            },
            'status': {
                'scale': 0.7, 'color': self.color_manager.get_color('ui', 'text_secondary'),
                'font': 'VeraMono.ttf', 'has_background': False,
            },
            'counter': {
                'scale': 0.7, 'color': color.yellow,
                'font': 'VeraMono.ttf', 'has_background': True,
            },
            'debug': {
                'scale': 0.7, 'color': color.cyan,
                'font': 'VeraMono.ttf', 'has_background': True,
            }
        }
    
    def create_element(self, category: str, name: str, 
                      text: str = "", position: tuple = (0, 0), 
                      style: str = 'default', **kwargs) -> Text:
        
        style_info = self.styles.get(style, self.styles['default']).copy()
        
        # Получаем флаг для фона и удаляем его, чтобы он не попал в конструктор
        should_have_background: bool = style_info.pop('has_background', False)
        
        element_kwargs: Dict[str, Any] = style_info
        element_kwargs.update(kwargs)
        element_kwargs['text'] = text
        element_kwargs['position'] = position
        
        # 1. Создаем объект Text без фона
        element = Text(**element_kwargs)
        element.style = style
        
        # 2. Включаем фон отдельной командой, как вы и предложили
        if should_have_background:
            element.background = True
        
        self.elements[category][name] = element
        return element
    
    def update_text(self, name: str, text: str) -> None:
        """Обновляет текст и принудительно перерисовывает фон."""
        if name in self.elements['dynamic']:
            element = self.elements['dynamic'][name]
            element.text = text
            
            # При обновлении текста также принудительно перерисовываем фон
            style_info = self.styles.get(element.style, self.styles['default'])
            if style_info.get('has_background'):
                element.background = True
    
    def get_element(self, name: str) -> Optional[Text]:
        for category in self.elements.values():
            if name in category:
                return category[name]
        return None
    
    def show_element(self, name: str) -> None:
        element = self.get_element(name)
        if element:
            element.enabled = True
    
    def hide_element(self, name: str) -> None:
        element = self.get_element(name)
        if element:
            element.enabled = False
    
    def toggle_element(self, name: str) -> None:
        element = self.get_element(name)
        if element:
            element.enabled = not element.enabled
    
    def show_category(self, category: str) -> None:
        if category in self.elements:
            for element in self.elements[category].values():
                element.enabled = True
    
    def hide_category(self, category: str) -> None:
        if category in self.elements:
            for element in self.elements[category].values():
                element.enabled = False
    
    def toggle_category(self, category: str) -> None:
        if category in self.elements:
            elements = list(self.elements[category].values())
            if elements:
                new_state = not elements[0].enabled
                for element in elements:
                    element.enabled = new_state
    
    def register_update_function(self, key: str, func: Callable[[], None]) -> None:
        self.update_functions[key] = func
    
    def update_dynamic_elements(self) -> None:
        for key, func in self.update_functions.items():
            try:
                func()
            except Exception as e:
                print(f"Ошибка обновления UI элемента {key}: {e}")
    
    def create_position_info(self, name: str = 'main', position: tuple = UI_POSITIONS.POSITION_INFO) -> Text:
        return self.create_element('dynamic', name,
            text="Position: 0.000, 0.000, 0.000\nRotation: 0.000, 0.000, 0.000",
            position=position, style='status')
    
    def create_instructions(self, name: str, instructions_text: str, position: tuple = UI_POSITIONS.DEFAULT_INSTRUCTIONS) -> Text:
        return self.create_element('static', name,
            text=instructions_text, position=position, style='instructions')
    
    def create_status_indicator(self, name: str, initial_text: str = "", position: tuple = UI_POSITIONS.CURSOR_STATUS) -> Text:
        return self.create_element('dynamic', name,
            text=initial_text, position=position, style='status', origin=(0, 0))
    
    def create_counter(self, name: str, initial_value: int = 0, position: tuple = UI_POSITIONS.DEFAULT_COUNTER, prefix: str = "") -> Text:
        text = f"{prefix}{initial_value}" if prefix else str(initial_value)
        return self.create_element('dynamic', name, text=text, position=position, style='counter')
    
    def update_counter(self, name: str, value: int, prefix: str = "") -> None:
        text = f"{prefix}{value}" if prefix else str(value)
        self.update_text(name, text)
    
    def create_info_block(self, name: str, title: str, content: str = "", position: tuple = UI_POSITIONS.DEFAULT_INFO_BLOCK, is_dynamic: bool = False) -> Text:
        category = 'dynamic' if is_dynamic else 'static'
        text = f"{title}\n{content}" if content else title
        return self.create_element(category, name, text=text, position=position, style='default')
    
    def update_info_block(self, name: str, title: str, content: str = "") -> None:
        text = f"{title}\n{content}" if content else title
        self.update_text(name, text)
    
    def create_debug_info(self, name: str, position: tuple = UI_POSITIONS.DEFAULT_DEBUG_INFO) -> Text:
        return self.create_element('dynamic', name, text="", position=position, style='debug')
    
    def remove_element(self, name: str) -> None:
        for category_elements in self.elements.values():
            if name in category_elements:
                destroy(category_elements[name])
                del category_elements[name]
                return
    
    def clear_category(self, category: str) -> None:
        if category in self.elements:
            for element in list(self.elements[category].values()):
                destroy(element)
            self.elements[category].clear()
    
    def clear_all(self) -> None:
        for category in list(self.elements.keys()):
            self.clear_category(category)
    
    def get_stats(self) -> Dict[str, int]:
        return {'static': len(self.elements['static']), 'dynamic': len(self.elements['dynamic']),
                'total': len(self.elements['static']) + len(self.elements['dynamic'])}
    
    def print_stats(self) -> None:
        stats = self.get_stats()
        print("\n--- UI Manager Stats ---")
        for category, count in stats.items():
            print(f"  {category.capitalize()}: {count}")
        print("------------------------")

def get_ui_manager(color_manager: Optional[ColorManager] = None) -> UIManager:
    return UIManager(color_manager)
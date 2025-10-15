"""
Controls Window - Автогенерируемое окно с инструкциями управления
================================================================

Создает окно с командами управления, автоматически генерируя
содержимое из зарегистрированных команд InputManager.
"""

from ursina import Text, color
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..managers.input_manager import InputManager
    from ..managers.color_manager import ColorManager


class ControlsWindow:
    """
    Окно с автогенерируемыми инструкциями управления.
    
    Автоматически создает список команд из InputManager
    и отображает их в удобном виде с группировкой по категориям.
    """
    
    # Key mapping for clean display
    KEY_DISPLAY_NAMES = {
        'scroll up': 'Scroll Up',
        'scroll down': 'Scroll Down',
        'wheel up': 'Scroll Up',
        'wheel down': 'Scroll Down',
        'left mouse down': 'LMB',
        'right mouse down': 'RMB',
        'space': 'Space',
        'shift': 'Shift',
        'escape': 'Esc',
        'ctrl+z': 'Ctrl+Z',
        'ctrl+c': 'Ctrl+C',
        'ctrl+scroll': 'Ctrl+Scroll',
        'ctrl+shift+c': 'Ctrl+Shift+C',
        'ctrl+i': 'Ctrl+I',
        'ctrl+shift+i': 'Ctrl+Shift+I',
        'alt': 'Alt',
        ':': ':',
        ';': ';',
        ',': ',',
        '[': '[',
        ']': ']',
    }
    
    # Category order and display names
    CATEGORY_ORDER = [
        ('zoom', 'ZOOM'),
        ('candidates', 'CANDIDATES'),
        ('visual', 'VISUAL'),
        ('time', 'TIME'),
        ('merge', 'MERGE'),
        ('optimize', 'OPTIMIZE'),
        ('tree', 'TREE'),
        ('ghosts', 'GHOSTS'),
        ('debug', 'DEBUG'),
        ('delete', 'DELETE'),
        ('stats', 'STATS'),
        ('ui', 'UI'),
    ]
    
    def __init__(self, 
                 input_manager: 'InputManager',
                 color_manager: Optional['ColorManager'] = None,
                 position: Tuple[float, float] = (-0.3, 0.1),
                 scale: float = 0.7):
        """
        Инициализация окна управления.
        
        Args:
            input_manager: Менеджер команд для получения списка
            color_manager: Менеджер цветов для стилизации
            position: Позиция окна на экране (нормализованные координаты)
            scale: Масштаб текста
        """
        self.input_manager = input_manager
        self.color_manager = color_manager
        try:
            self.position = (float(position[0]), float(position[1]))
        except (TypeError, AttributeError):
            self.position = position
        self.scale = scale
        self.visible = True
        
        # UI elements
        self.window_text = None  # Объединенный текст (заголовок + команды)
        
        # Кеш для сгенерированного текста
        self._cached_text = ""
        self._last_commands_hash = None
        
        # Создаем окно
        self._create_window()
        self._update_controls_text()
    
    def _create_window(self):
        """Создает визуальные элементы окна."""
        # Получаем цвет текста
        text_color = color.light_gray
        if self.color_manager:
            try:
                text_color = self.color_manager.get_color('ui', 'text_primary')
            except:
                pass
                
        # Создаем единый Text элемент для всего окна (заголовок + команды)
        self.window_text = Text(
            'CONTROLS\n\n',  # Заголовок + пустая строка, команды добавятся в _update_controls_text
            position=(self.position[0] - 0.15, self.position[1] + 0.23),
            scale=self.scale,
            color=text_color,
            font='VeraMono.ttf',
            enabled=self.visible
        )
        # Включаем фон отдельно (как в ui_manager)
        self.window_text.background = True
    
    def _format_key(self, key: str) -> str:
        """
        Форматирует клавишу для отображения.
        
        Args:
            key: Название клавиши из InputManager
            
        Returns:
            Отформатированное название
        """
        # Используем маппинг если есть
        if key in self.KEY_DISPLAY_NAMES:
            return self.KEY_DISPLAY_NAMES[key]
        
        # Для букв и цифр - просто uppercase
        if len(key) == 1:
            return key.upper()
        
        # Для остального - capitalize
        return key.replace('_', ' ').title()
    
    
    def _group_similar_commands(self, commands: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Группирует похожие команды для компактного отображения.
        
        Например: W, A, S, D -> WASD - move camera
        """
        # Группируем команды движения
        movement_keys = []
        movement_desc = ""
        zoom_keys = []
        zoom_desc = ""
        scale_keys = []
        scale_desc = ""
        
        grouped = []
        skip_keys = set()
        
        for key, desc in commands:
            if key in skip_keys:
                continue
                
            # Zoom grouping
            if key in ['e', 't'] and 'zoom' in desc.lower():
                zoom_keys.append(self._format_key(key))
                zoom_desc = "zoom in/out"
                skip_keys.update(['e', 't'])
            
            # Spore scale grouping
            elif key in ['1', '2'] and 'scale' in desc.lower():
                scale_keys.append(self._format_key(key))
                scale_desc = "scale spores"
                skip_keys.update(['1', '2'])
            
            # Остальные команды добавляем как есть
            elif key not in skip_keys:
                grouped.append((self._format_key(key), desc))
        
        # Add grouped commands
        if zoom_keys:
            grouped.insert(0, ("E/T", zoom_desc))
        if scale_keys:
            idx = 1 if zoom_keys else 0
            grouped.insert(idx, ("1/2", scale_desc))
        
        return grouped
    
    def _generate_controls_text(self) -> str:
        """
        Генерирует текст с командами из InputManager.
        
        Returns:
            Отформатированный текст для отображения
        """
        if not self.input_manager or not hasattr(self.input_manager, 'get_commands_by_category'):
            return "Commands not available"
        
        lines = []
        commands_by_category = self.input_manager.get_commands_by_category()
        
        # Проходим по категориям в заданном порядке
        for cat_key, cat_display in self.CATEGORY_ORDER:
            if cat_key not in commands_by_category:
                continue
            
            commands = commands_by_category[cat_key]
            if not commands:
                continue
            
            # Фильтруем только включенные команды
            enabled_commands = []
            for key, desc in commands:
                if key in self.input_manager.commands:
                    cmd_info = self.input_manager.commands[key]
                    if cmd_info['enabled']():
                        # Use description as-is (already in English)
                        enabled_commands.append((key, desc))
            
            if not enabled_commands:
                continue
            
            # Group similar commands for zoom category
            if cat_key == 'zoom':
                enabled_commands = self._group_similar_commands(enabled_commands)
            
            # Add category commands
            for key, desc in enabled_commands:
                formatted_key = self._format_key(key) if cat_key != 'zoom' else key
                # Format line with alignment
                lines.append(f"{formatted_key:<12} - {desc}")
        
        return '\n'.join(lines)
    
    def _update_controls_text(self):
        """Refresh cached controls text from the input manager."""
        commands_text = self._generate_controls_text()
        
        # Объединяем заголовок и команды
        full_text = f'CONTROLS\n\n{commands_text}'
        
        if full_text != self._cached_text:
            self._cached_text = full_text
            if self.window_text:
                self.window_text.text = full_text
                # Принудительно обновляем фон при изменении текста
                self.window_text.background = True
    
    def set_visibility(self, visible: bool):
        """Set window visibility explicitly."""
        self.visible = visible
        
        if self.window_text:
            self.window_text.enabled = self.visible
    
    def toggle_visibility(self):
        """Toggle window visibility."""
        self.set_visibility(not self.visible)
        
        status = "shown" if self.visible else "hidden"
        print(f"[ControlsWindow] {status}")
    
    def update_commands(self):
        """Update displayed commands (e.g. after bindings change)."""
        self._update_controls_text()
    def set_position(self, position: Tuple[float, float]):
        """Изменяет позицию окна."""
        self.position = position
        
        if self.window_text:
            self.window_text.position = (position[0] - 0.15, position[1] + 0.23)

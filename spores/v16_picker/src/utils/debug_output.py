"""
Утилита для управления отладочным выводом
========================================

Позволяет централизованно управлять различными типами отладочного вывода
через конфигурацию без изменения кода.
"""

from typing import Dict, Optional, Any


class DebugOutput:
    """Централизованное управление отладочным выводом."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Инициализирует систему отладочного вывода.
        
        Args:
            config: Словарь конфигурации с секцией 'debug'
        """
        self.config = config if config is not None else {}
        debug_config = self.config.get('debug', {})
        
        # Флаги для различных типов отладки
        self.verbose_output = debug_config.get('enable_verbose_output', False)
        self.detailed_evolution = debug_config.get('enable_detailed_evolution', False)
        self.candidate_logging = debug_config.get('enable_candidate_logging', False)
        self.trajectory_logging = debug_config.get('enable_trajectory_logging', False)
        
        # Совместимость со старым флагом trajectory_optimization.debug_output
        old_debug = self.config.get('trajectory_optimization', {}).get('debug_output', False)
        if old_debug:
            self.trajectory_logging = True
    
    def print_verbose(self, *args, **kwargs) -> None:
        """Выводит общие отладочные сообщения."""
        if self.verbose_output:
            print(*args, **kwargs)
    
    def print_evolution(self, *args, **kwargs) -> None:
        """Выводит детальную информацию об эволюции спор."""
        if self.detailed_evolution:
            print(*args, **kwargs)
    
    def print_candidate(self, *args, **kwargs) -> None:
        """Выводит информацию о кандидатских спорах."""
        if self.candidate_logging:
            print(*args, **kwargs)
    
    def print_trajectory(self, *args, **kwargs) -> None:
        """Выводит информацию о траекториях и оптимизации."""
        if self.trajectory_logging:
            print(*args, **kwargs)
    
    def print_always(self, *args, **kwargs) -> None:
        """Выводит важные сообщения всегда (ошибки, предупреждения)."""
        print(*args, **kwargs)
    
    def is_verbose_enabled(self) -> bool:
        """Проверяет включен ли общий отладочный вывод."""
        return self.verbose_output
    
    def is_evolution_enabled(self) -> bool:
        """Проверяет включен ли отладочный вывод эволюции."""
        return self.detailed_evolution
    
    def is_candidate_enabled(self) -> bool:
        """Проверяет включен ли отладочный вывод кандидатов."""
        return self.candidate_logging
    
    def is_trajectory_enabled(self) -> bool:
        """Проверяет включен ли отладочный вывод траекторий."""
        return self.trajectory_logging


# Глобальный экземпляр (будет инициализирован позже)
_debug_output: Optional[DebugOutput] = None


def init_debug_output(config: Optional[Dict] = None) -> None:
    """Инициализирует глобальный экземпляр отладочного вывода."""
    global _debug_output
    _debug_output = DebugOutput(config)


def get_debug_output() -> DebugOutput:
    """Возвращает глобальный экземпляр отладочного вывода."""
    global _debug_output
    if _debug_output is None:
        # Создаем с пустой конфигурацией (всё выключено)
        _debug_output = DebugOutput({})
    return _debug_output


# Удобные функции для быстрого доступа
def debug_print(*args, **kwargs) -> None:
    """Общий отладочный вывод."""
    get_debug_output().print_verbose(*args, **kwargs)


def evolution_print(*args, **kwargs) -> None:
    """Отладочный вывод эволюции."""
    get_debug_output().print_evolution(*args, **kwargs)


def candidate_print(*args, **kwargs) -> None:
    """Отладочный вывод кандидатов."""
    get_debug_output().print_candidate(*args, **kwargs)


def trajectory_print(*args, **kwargs) -> None:
    """Отладочный вывод траекторий."""
    get_debug_output().print_trajectory(*args, **kwargs)


def always_print(*args, **kwargs) -> None:
    """Важные сообщения (всегда)."""
    get_debug_output().print_always(*args, **kwargs) 
"""
nb_logger - Notebook output logger
====================================

Usage (first cell of notebook):
    from src.nb_logger import start
    start()
"""

import sys
from pathlib import Path
from datetime import datetime

# Сохраняем оригинальный stdout ДО любых замен (при импорте модуля)
_real_stdout = sys.stdout
_real_stderr = sys.stderr
_real_displayhook = sys.displayhook
_registered_hook = [None]
_log_path = [None]


class _Tee:
    def __init__(self, log_path: Path, real_stream):
        self._log_path = log_path
        self._real = real_stream

    def write(self, s: str) -> None:
        self._real.write(s)
        with open(self._log_path, 'a', encoding='utf-8') as f:
            f.write(s)

    def flush(self) -> None:
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


def start(path=None):
    global _real_stdout, _real_stderr

    if path is None:
        frame = sys._getframe(1)
        nb_path = frame.f_globals.get('__vsc_ipynb_file__')
        path = Path(nb_path).with_suffix('.txt') if nb_path else Path('notebook.txt')

    path = Path(path)
    _log_path[0] = path
    path.write_text(
        f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 60}\n",
        encoding='utf-8'
    )

    # Не наматываем Tee поверх Tee при повторном вызове
    if not isinstance(sys.stdout, _Tee):
        _real_stdout = sys.stdout
    if not isinstance(sys.stderr, _Tee):
        _real_stderr = sys.stderr

    sys.stdout = _Tee(path, _real_stdout)
    sys.stderr = _Tee(path, _real_stderr)

    # Перехватываем displayhook (вывод последнего выражения ячейки без print)
    def _displayhook(value):
        _real_displayhook(value)
        if value is not None:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(repr(value) + '\n')

    sys.displayhook = _displayhook

    # Регистрируем хук
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return

        if _registered_hook[0] is not None:
            try:
                ip.events.unregister('pre_execute', _registered_hook[0])
            except ValueError:
                pass

        _cell_n = [0]

        def _pre_execute():
            _cell_n[0] += 1
            with open(_log_path[0], 'a', encoding='utf-8') as f:
                f.write(f'\n{"─" * 50}\nCell {_cell_n[0]}\n{"─" * 50}\n')

        ip.events.register('pre_execute', _pre_execute)
        _registered_hook[0] = _pre_execute

    except ImportError:
        pass

    print(f"[nb_logger] Logging to: {path}")


def stop():
    sys.stdout = _real_stdout
    sys.stderr = _real_stderr
    sys.displayhook = _real_displayhook
    _log_path[0] = None
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip and _registered_hook[0]:
            ip.events.unregister('pre_execute', _registered_hook[0])
            _registered_hook[0] = None
    except Exception:
        pass

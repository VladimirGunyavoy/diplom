"""
ParamManager - Named simulation parameters
==========================================

Stores named float parameters. Knows nothing about keys or input.
Each param can be tweaked exponentially or linearly, with optional min/max clamp.
"""


class _Param:
    def __init__(self, value: float, mode: str, factor: float, step: float, min_val, max_val):
        self.mode = mode
        self.factor = factor
        self.step = step
        self.value = value
        self.min_val = min_val
        self.max_val = max_val


class ParamManager:

    def __init__(self):
        self._params: dict[str, _Param] = {}

    def add(self, name: str, value: float, mode: str = 'exp', factor: float = 1.05, step: float = 0.1,
            min_val=None, max_val=None) -> None:
        """Register a named parameter with initial value and tweak settings."""
        self._params[name] = _Param(value, mode, factor, step, min_val, max_val)
        setattr(self, name, value)

    def tweak(self, name: str, sign: int) -> None:
        """Change parameter by one step. sign: +1 increase, -1 decrease."""
        param = self._params[name]
        current = getattr(self, name)
        if param.mode == 'exp':
            new_val = current * (param.factor ** sign)
        else:
            new_val = current + sign * param.step
        if param.min_val is not None:
            new_val = max(new_val, param.min_val)
        if param.max_val is not None:
            new_val = min(new_val, param.max_val)
        setattr(self, name, new_val)
        param.value = new_val
        print(f"[ParamManager] {name}: {new_val:.4f}")

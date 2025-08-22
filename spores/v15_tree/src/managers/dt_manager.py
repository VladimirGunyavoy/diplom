# 🔧 ИСПРАВЛЕННЫЙ DTManager с правильным reset

class DTManager:
    """
    Менеджер для интерактивного управления временным шагом (dt).
    """
    
    def __init__(self, config: dict, pendulum_system: 'PendulumSystem'):
        self.config = config
        self.pendulum = pendulum_system
        
        # 🆕 ИСПРАВЛЕНИЕ: Сохраняем оригинальное значение dt ПРИ ИНИЦИАЛИЗАЦИИ
        self.original_dt = float(config.get('pendulum', {}).get('dt', 0.1))
        self.current_dt = self.original_dt  # Начинаем с оригинального
        
        # Настройки для изменения dt
        self.dt_multiplier = 1.1  # На сколько изменяем dt за один шаг колесика
        self.min_dt = 0.001       # Минимальное значение dt
        self.max_dt = 1.0         # Максимальное значение dt
        
        print(f"   ✓ DTManager создан (начальный dt: {self.current_dt})")
        print(f"   📋 Оригинальный dt сохранен: {self.original_dt}")

    def increase_dt(self) -> float:
        """Увеличивает dt мультипликативно."""
        old_dt = self.current_dt
        self.current_dt = min(self.current_dt * self.dt_multiplier, self.max_dt)
        
        if self.current_dt != old_dt:
            self._update_config()
            print(f"   🔼 dt увеличен: {old_dt:.4f} → {self.current_dt:.4f} (×{self.dt_multiplier})")
        else:
            print(f"   ⚠️ dt уже максимальный: {self.current_dt:.4f}")
            
        return self.current_dt

    def decrease_dt(self) -> float:
        """Уменьшает dt мультипликативно."""
        old_dt = self.current_dt
        self.current_dt = max(self.current_dt / self.dt_multiplier, self.min_dt)
        
        if self.current_dt != old_dt:
            self._update_config()
            print(f"   🔽 dt уменьшен: {old_dt:.4f} → {self.current_dt:.4f} (÷{self.dt_multiplier})")
        else:
            print(f"   ⚠️ dt уже минимальный: {self.current_dt:.4f}")
            
        return self.current_dt

    def reset_dt(self) -> float:
        """
        🔧 ИСПРАВЛЕНО: Сбрасывает dt к СОХРАНЕННОМУ исходному значению.
        
        Returns:
            Сброшенное значение dt
        """
        old_dt = self.current_dt
        self.current_dt = self.original_dt  # 🆕 Используем сохраненное значение!
        
        self._update_config()
        print(f"   🔄 dt сброшен: {old_dt:.4f} → {self.current_dt:.4f} (оригинал: {self.original_dt})")
        
        return self.current_dt

    def get_dt(self) -> float:
        """Возвращает текущее значение dt."""
        return self.current_dt

    def set_dt(self, new_dt: float) -> float:
        """
        Устанавливает новое значение dt с проверкой границ.
        
        Args:
            new_dt: Новое значение dt
            
        Returns:
            Фактически установленное значение dt
        """
        old_dt = self.current_dt
        self.current_dt = max(self.min_dt, min(new_dt, self.max_dt))
        
        if self.current_dt != old_dt:
            self._update_config()
            print(f"   ⚙️ dt установлен: {old_dt:.4f} → {self.current_dt:.4f}")
            
        return self.current_dt

    def _update_config(self) -> None:
        """Обновляет значение dt в конфигурации."""
        if 'pendulum' not in self.config:
            self.config['pendulum'] = {}
        self.config['pendulum']['dt'] = self.current_dt
        

    def get_stats(self) -> dict:
        """Возвращает статистику dt менеджера."""
        # 🆕 ИСПРАВЛЕНИЕ: Используем сохраненный original_dt
        multiplier_from_original = self.current_dt / self.original_dt if self.original_dt != 0 else 1.0
        
        return {
            'current_dt': self.current_dt,
            'original_dt': self.original_dt,  # 🆕 Из сохраненного значения
            'multiplier_from_original': multiplier_from_original,
            'min_dt': self.min_dt,
            'max_dt': self.max_dt,
            'dt_multiplier': self.dt_multiplier,
            'at_min': self.current_dt <= self.min_dt,
            'at_max': self.current_dt >= self.max_dt
        }

    def print_stats(self) -> None:
        """Выводит подробную статистику dt."""
        stats = self.get_stats()
        
        print(f"\n⏱️ СТАТИСТИКА DT:")
        print(f"   📊 Текущий dt: {stats['current_dt']:.4f}")
        print(f"   📋 Оригинальный dt: {stats['original_dt']:.4f}")
        print(f"   📈 Изменение: ×{stats['multiplier_from_original']:.2f}")
        print(f"   ⬇️ Минимум: {stats['min_dt']:.4f} {'(достигнут)' if stats['at_min'] else ''}")
        print(f"   ⬆️ Максимум: {stats['max_dt']:.4f} {'(достигнут)' if stats['at_max'] else ''}")
        print(f"   🔧 Множитель: ×÷{stats['dt_multiplier']}")
        print("========================")

    # 🆕 ДОПОЛНИТЕЛЬНЫЙ МЕТОД для отладки
    def debug_values(self) -> None:
        """Выводит все значения для отладки."""
        print(f"\n🐛 ОТЛАДКА DTManager:")
        print(f"   original_dt (сохранен при __init__): {self.original_dt}")
        print(f"   current_dt: {self.current_dt}")
        print(f"   config['pendulum']['dt']: {self.config.get('pendulum', {}).get('dt', 'НЕТ')}")
        print(f"   Одинаковы ли current и config: {self.current_dt == self.config.get('pendulum', {}).get('dt', -1)}")
        print("========================")
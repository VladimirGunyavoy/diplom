# 🔧 ИСПРАВЛЕННЫЙ DTManager с правильным reset

class DTManager:
    """
    Менеджер для интерактивного управления временным шагом (dt).
    """
    
    def __init__(self, config: dict, pendulum_system: 'PendulumSystem'):
        self.config = config
        self.pendulum = pendulum_system
        
        # Связь с менеджером спор будет проставлена снаружи (см. main_demo.py)
        self.spore_manager = None  # type: ignore

        # Параметр для масштабирования допустимой длины линка от dt:
        # config['link']['distance_per_dt'] — опционален; по умолчанию 1.0
        link_cfg = config.get('link', {}) if isinstance(config, dict) else {}
        self.link_length_per_dt = float(link_cfg.get('distance_per_dt', 1.0))
        
        # 🆕 ИСПРАВЛЕНИЕ: Сохраняем оригинальное значение dt ПРИ ИНИЦИАЛИЗАЦИИ
        self.original_dt = float(config.get('pendulum', {}).get('dt', 0.1))
        self.current_dt = self.original_dt  # Начинаем с оригинального
        
        # Настройки для изменения dt
        self.dt_multiplier = 1.1  # На сколько изменяем dt за один шаг колесика
        self.min_dt = 0.001       # Минимальное значение dt
        self.max_dt = 1.0         # Максимальное значение dt
        
        # Система подписок на изменения dt
        self._subscribers = []
        
        print(f"   ✓ DTManager создан (начальный dt: {self.current_dt})")
        print(f"   📋 Оригинальный dt сохранен: {self.original_dt}")

    def increase_dt(self) -> float:
        """Увеличивает dt мультипликативно."""
        old = float(self.current_dt)
        self.current_dt = min(self.current_dt * self.dt_multiplier, self.max_dt)
        
        if self.current_dt != old:
            self._update_config()
            print(f"   🔼 dt увеличен: {old:.4f} → {self.current_dt:.4f} (×1.1)")
            self._notify_dt_changed()  # ← ОБЯЗАТЕЛЬНО
        else:
            print(f"   ⚠️ dt уже максимальный: {self.current_dt:.4f}")
            
        return self.current_dt

    def decrease_dt(self) -> float:
        """Уменьшает dt мультипликативно."""
        old = float(self.current_dt)
        self.current_dt = max(self.current_dt / self.dt_multiplier, self.min_dt)
        
        if self.current_dt != old:
            self._update_config()
            print(f"   🔽 dt уменьшен: {old:.4f} → {self.current_dt:.4f} (÷1.1)")
            self._notify_dt_changed()  # ← ОБЯЗАТЕЛЬНО
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
        self._notify_dt_changed()
        
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
            self._notify_dt_changed()
            
        return self.current_dt

    def _update_config(self) -> None:
        """Обновляет значение dt в конфигурации."""
        if 'pendulum' not in self.config:
            self.config['pendulum'] = {}
        self.config['pendulum']['dt'] = self.current_dt

    def get_max_link_length(self) -> float:
        """Максимально допустимая длина линка в текущих координатах сцены."""
        return float(self.current_dt * self.link_length_per_dt)

    def _notify_dt_changed(self) -> None:
        """Колбэк при изменении dt — обновляем длины всех линков и уведомляем подписчиков."""
        try:
            if getattr(self, 'spore_manager', None):
                print(f"[DT] applying max_length to SporeManager.links (max_len={self.get_max_link_length():.6f})")
                self.spore_manager.update_links_max_length(self.get_max_link_length())
        except Exception as e:
            print(f"[DTManager] Ошибка при обновлении длин линков: {e}")
        
        # Уведомляем всех подписчиков
        self._notify_subscribers()

    def subscribe_on_change(self, callback) -> None:
        """Подписывает колбэк на изменения dt."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)
            print(f"   📧 Подписчик добавлен в DTManager (всего: {len(self._subscribers)})")

    def unsubscribe_on_change(self, callback) -> None:
        """Отписывает колбэк от изменений dt."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            print(f"   📧 Подписчик удален из DTManager (осталось: {len(self._subscribers)})")

    def _notify_subscribers(self) -> None:
        """Уведомляет всех подписчиков об изменении dt."""
        n = len(getattr(self, "_subscribers", []))
        print(f"[DT] notifying {n} subscriber(s)  [DT id={id(self)}]")
        for i, cb in enumerate(self._subscribers):
            print(f"[DT] -> subscriber[{i}] {getattr(cb, '__name__', str(cb))}")
            cb()

    def debug_subscribers(self) -> None:
        """Выводит список всех подписчиков для отладки."""
        print(f"[DT] subscribers dump (id={id(self)}):")
        for i, cb in enumerate(self._subscribers):
            print(f"   [{i}] {getattr(cb, '__name__', str(cb))}")

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
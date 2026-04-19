#!/usr/bin/env python3
"""
Простой тест для проверки работы флага DEBUG_PM_SPAM в PredictionManager.
"""

def test_debug_pm_spam_flag():
    """Тестирует работу флага DEBUG_PM_SPAM."""
    
    print("🧪 Тестирование флага DEBUG_PM_SPAM в PredictionManager")
    print("=" * 60)
    
    # Читаем файл напрямую
    try:
        with open('src/managers/manual_creation/prediction_manager.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем наличие флага
        if 'DEBUG_PM_SPAM = False' in content:
            print("✅ Флаг DEBUG_PM_SPAM найден и установлен в False")
        else:
            print("❌ Флаг DEBUG_PM_SPAM не найден или имеет неожиданное значение")
            return False
        
        # Проверяем обернутые логи
        wrapped_logs = [
            'if DEBUG_PM_SPAM: print(f"[PM] update_predictions',
            'if DEBUG_PM_SPAM: print("[PM] _update_tree_preview: cleared old predictions")',
            'if DEBUG_PM_SPAM: print(f"[PredictionManager] ghost_dt_vector head:',
            'if DEBUG_PM_SPAM: print(f"[PM] +ghost_link',
            'if DEBUG_PM_SPAM: print(f"[PM] clear_predictions: removing',
            'if DEBUG_PM_SPAM: print("[PM] clear_predictions: done")',
            'if DEBUG_PM_SPAM: print(f"[PM] _get_current_dt ->'
        ]
        
        print("\n📋 Проверка обернутых логов:")
        for log in wrapped_logs:
            if log in content:
                print(f"✅ {log[:50]}...")
            else:
                print(f"❌ {log[:50]}... (НЕ НАЙДЕН)")
                return False
        
        print("\n✅ Все логи успешно обернуты флагом DEBUG_PM_SPAM!")
        print("💡 Для включения спам-логов измените DEBUG_PM_SPAM = True в prediction_manager.py")
        
        return True
        
    except FileNotFoundError:
        print("❌ Файл prediction_manager.py не найден")
        return False
    except Exception as e:
        print(f"❌ Ошибка при чтении файла: {e}")
        return False

if __name__ == "__main__":
    success = test_debug_pm_spam_flag()
    exit(0 if success else 1)

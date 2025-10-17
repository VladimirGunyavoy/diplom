"""
Тестовый скрипт для проверки valence_manager
"""
import sys
sys.path.insert(0, 'spores/v16_picker/src')

# Минимальная проверка что код компилируется
try:
    from logic.valence import SporeValence, ValenceSlot
    print("OK: Modules imported successfully")

    # Проверяем что слоты детей в правильном порядке
    valence = SporeValence(spore_id="test")
    print("\nOK: Children slot order:")
    for i, slot in enumerate(valence.children_slots):
        print(f"  {i+1}. {slot.get_slot_name()}")

    expected_order = ['forward_max', 'forward_min', 'backward_max', 'backward_min']
    actual_order = [slot.get_slot_name() for slot in valence.children_slots]

    if actual_order == expected_order:
        print("  OK: Order is correct!")
    else:
        print(f"  ERROR! Expected {expected_order}, got {actual_order}")

    # Проверяем что внуков 8 (не 4)
    print(f"\nOK: Number of grandchildren slots: {len(valence.grandchildren_slots)}")
    if len(valence.grandchildren_slots) == 8:
        print("  OK: Correct (8 slots with alternation)")
    else:
        print(f"  ERROR! Should be 8, not {len(valence.grandchildren_slots)}")

    print("\nOK: All grandchildren slots (with control alternation):")
    for i, slot in enumerate(valence.grandchildren_slots):
        print(f"  {i+1}. {slot.get_slot_name()}")

    # Проверяем что есть все 8 ожидаемых маршрутов
    expected_grandchildren = [
        'forward_max_forward_min',
        'forward_max_backward_min',
        'forward_min_forward_max',
        'forward_min_backward_max',
        'backward_max_forward_min',
        'backward_max_backward_min',
        'backward_min_forward_max',
        'backward_min_backward_max',
    ]

    actual_grandchildren = [slot.get_slot_name() for slot in valence.grandchildren_slots]

    print("\nOK: Checking all expected routes:")
    all_present = True
    for expected in expected_grandchildren:
        if expected in actual_grandchildren:
            print(f"  OK: {expected}")
        else:
            print(f"  MISSING: {expected}")
            all_present = False

    if all_present:
        print("\n=== ALL TESTS PASSED ===")
    else:
        print("\n=== ERRORS FOUND ===")

except Exception as e:
    print(f"ERROR during import or testing: {e}")
    import traceback
    traceback.print_exc()

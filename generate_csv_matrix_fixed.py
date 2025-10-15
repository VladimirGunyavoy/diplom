#!/usr/bin/env python3
"""
Генератор CSV матрицы связей между спорами из JSON данных.

ПРАВИЛЬНАЯ ЛОГИКА:
- Управление (control) остается тем же для обратной связи
- Время (dt) инвертируется для обратной связи

Пример:
- Прямая связь A→B: control=+2.0, dt=+0.05
- Обратная связь B→A: control=+2.0, dt=-0.05
"""

import json
import csv
import os
from typing import Dict, List, Tuple, Optional


def load_json_data(json_path: str) -> Optional[dict]:
    """Загружает данные из JSON файла."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Ошибка загрузки JSON: {e}")
        return None


def extract_spores_info(graph_data: dict) -> List[Dict]:
    """Извлекает информацию о спорах."""
    spores = graph_data.get('spores', [])
    print(f"📊 Найдено спор: {len(spores)}")
    return spores


def extract_links_info(graph_data: dict) -> List[Dict]:
    """Извлекает информацию о связях."""
    links = graph_data.get('links', [])
    print(f"🔗 Найдено связей: {len(links)}")
    return links


def build_links_matrix(spores: List[Dict], links: List[Dict]) -> Tuple[List[str], List[List[str]]]:
    """
    Строит матрицу связей между спорами.
    Показывает ВСЕ связи + их дубликаты с обратным направлением.
    
    ЛОГИКА:
    - Берем связи из JSON
    - Для каждой связи создаем основную + дублированную с обратным направлением
    - Основная: исходное направление, корректированное время
    - Дубликат: обратное направление, инвертированное время
    
    Returns:
        Tuple[List[str], List[List[str]]]: (заголовки, матрица)
    """
    # Получаем все индексы спор и сортируем
    spore_indices = sorted([spore['index'] for spore in spores])
    max_index = max(spore_indices) if spore_indices else 0
    
    print(f"📈 Индексы спор: {spore_indices}")
    print(f"📈 Максимальный индекс: {max_index}")
    
    # Создаем словарь связей: (from_index, to_index) -> link_info
    links_dict = {}
    
    for link in links:
        parent_spore_id = link.get('parent_spore_id')
        child_spore_id = link.get('child_spore_id')
        
        # Используем parent_spore_id и child_spore_id как индексы напрямую
        # Проверяем, что индексы существуют в списке спор
        parent_index = None
        child_index = None
        
        # Ищем споры с соответствующими индексами
        spore_indices_set = {spore['index'] for spore in spores}
        
        # Пробуем разные варианты соответствия
        # Вариант 1: parent_spore_id как index напрямую
        if (parent_spore_id - 1) in spore_indices_set:
            parent_index = parent_spore_id - 1
        # Вариант 2: ищем по spore_id
        else:
            for spore in spores:
                if spore['spore_id'] == str(parent_spore_id):
                    parent_index = spore['index']
                    break
        
        # То же для child_spore_id
        if (child_spore_id - 1) in spore_indices_set:
            child_index = child_spore_id - 1
        else:
            for spore in spores:
                if spore['spore_id'] == str(child_spore_id):
                    child_index = spore['index']
                    break
        
        if parent_index is not None and child_index is not None:
            control = link.get('control', 0)
            dt = link.get('dt', 0)
            dt_sign = link.get('dt_sign', 1)
            
            # Используем raw_dt из JSON, но инвертируем для обратных связей
            raw_dt = link.get('raw_dt', dt * dt_sign)
            direction = link.get('direction', 'forward')
            
            # Для обратных связей инвертируем знак времени
            if direction == 'backward':
                dt_value = -raw_dt  # Инвертируем для получения положительного времени
            else:
                dt_value = raw_dt
            
            # Определяем базовые значения
            control_str = f"+{control}" if control >= 0 else str(control)
            
            # ✅ Показываем ВСЕ связи без фильтрации по времени
            # Основная связь: parent → child (время из JSON)
            dt_str = f"{dt_value:+.3f}"
            link_text = f"{control_str} {dt_str}"
            links_dict[(parent_index, child_index)] = link_text
            
            print(f"🔗 Связь: {parent_index+1} → {child_index+1} = '{link_text}' (время: {dt_value:+.3f}) [строка {parent_index+1}, столбец {child_index+1}]")
            
            # 🔄 Добавляем дублированную связь с обратным направлением, инвертированным только временем
            reverse_dt_str = f"{-dt_value:+.3f}"
            reverse_link_text = f"{control_str} {reverse_dt_str}"
            links_dict[(child_index, parent_index)] = reverse_link_text
            
            print(f"🔄 Дублируем обратно: {child_index+1} → {parent_index+1} = '{reverse_link_text}' (control: {control}, время: {-dt_value:+.3f}) [строка {child_index+1}, столбец {parent_index+1}]")
    
    # Создаем заголовки (индексы + 1 для отображения)
    headers = [''] + [str(idx + 1) for idx in spore_indices]
    
    # Создаем матрицу
    matrix = []
    for row_idx in spore_indices:
        row = [str(row_idx + 1)]  # Первый столбец - индекс строки
        
        for col_idx in spore_indices:
            # Проверяем есть ли связь от row_idx к col_idx
            link_text = links_dict.get((row_idx, col_idx), '')
            row.append(link_text)
        
        matrix.append(row)
    
    return headers, matrix


def save_csv_matrix(headers: List[str], matrix: List[List[str]], output_path: str):
    """Сохраняет матрицу в CSV файл."""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Записываем заголовки
            writer.writerow(headers)
            
            # Записываем строки матрицы
            writer.writerows(matrix)
        
        print(f"✅ CSV матрица сохранена: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка сохранения CSV: {e}")
        return False


def print_matrix_preview(headers: List[str], matrix: List[List[str]]):
    """Выводит превью матрицы в консоль."""
    print("\n📋 ПРЕВЬЮ CSV МАТРИЦЫ:")
    print("=" * 50)
    
    # Выводим заголовки
    header_line = " | ".join(f"{h:>8}" for h in headers)
    print(header_line)
    print("-" * len(header_line))
    
    # Выводим строки (ограничиваем для превью)
    for i, row in enumerate(matrix[:10]):  # Показываем только первые 10 строк
        row_line = " | ".join(f"{cell:>8}" for cell in row)
        print(row_line)
        
        if i >= 9 and len(matrix) > 10:
            print(f"... (еще {len(matrix) - 10} строк)")
            break


def main():
    """Основная функция."""
    print("🚀 ГЕНЕРАТОР CSV МАТРИЦЫ СВЯЗЕЙ СПОР")
    print("=" * 60)
    print("✅ ЛОГИКА: Все связи + дубликаты, без фильтрации")
    print("   - Строки = исходные споры (откуда)")  
    print("   - Столбцы = целевые споры (куда)")
    print("   - Ячейка [i,j] = связь i → j")
    print("=" * 60)
    
    # Путь к JSON файлу
    json_path = "spores/v16_picker/scripts/run/buffer/real_graph_latest.json"
    
    # Проверяем существование файла
    if not os.path.exists(json_path):
        print(f"❌ JSON файл не найден: {json_path}")
        return
    
    # Загружаем JSON данные
    print(f"📁 Загружаем: {json_path}")
    graph_data = load_json_data(json_path)
    if not graph_data:
        return
    
    # Извлекаем информацию
    spores = extract_spores_info(graph_data)
    links = extract_links_info(graph_data)
    
    if not spores:
        print("❌ Нет данных о спорах")
        return
    
    # Строим матрицу
    print("\n🔧 Построение матрицы связей...")
    headers, matrix = build_links_matrix(spores, links)
    
    # Показываем превью
    print_matrix_preview(headers, matrix)
    
    # Сохраняем CSV
    output_path = "spores/v16_picker/scripts/run/buffer/spores_links_matrix_fixed.csv"
    print(f"\n💾 Сохранение в: {output_path}")
    
    if save_csv_matrix(headers, matrix, output_path):
        print(f"\n🎉 ГОТОВО! CSV матрица создана: {output_path}")
        print(f"📊 Размер матрицы: {len(matrix)} x {len(headers)}")
        print(f"🔗 Связей найдено: {sum(1 for row in matrix for cell in row[1:] if cell.strip())}")
        print(f"\n✅ ПОКАЗАНЫ ВСЕ СВЯЗИ: прямые и обратные с любым временем!")
    else:
        print("\n❌ Ошибка при создании CSV файла")


if __name__ == "__main__":
    main()

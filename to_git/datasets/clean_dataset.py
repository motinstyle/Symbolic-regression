import os
import pandas as pd
import glob

def clean_csv_files():
    """
    Очищает CSV файлы от строк с отсутствующими значениями в текущей директории.
    """
    # Получаем текущую директорию (где находится скрипт)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Получаем список всех CSV файлов в директории
    csv_files = glob.glob(os.path.join(current_dir, '*.csv'))
    
    if not csv_files:
        print(f"В директории {current_dir} не найдено CSV файлов")
        return
    
    # Создаём директорию для резервных копий
    backup_dir = os.path.join(current_dir, 'backups')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"Создана директория для резервных копий: {backup_dir}")
    
    # Обрабатываем каждый CSV файл
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        
        print(f"\nОбработка файла: {file_name}")
        
        try:
            # Чтение CSV файла
            df = pd.read_csv(file_path)
            
            print(f"Исходное количество строк: {len(df)}")
            
            # Создаем резервную копию
            backup_path = os.path.join(backup_dir, file_name)
            df.to_csv(backup_path, index=False)
            print(f"Создана резервная копия: {backup_path}")
            
            # Подсчитываем количество строк с пропусками
            rows_with_na = df.isna().any(axis=1).sum()
            
            # Удаляем строки с пропусками в любом столбце
            df_cleaned = df.dropna()
            
            # Выводим статистику
            print(f"Удалено строк с пропусками: {rows_with_na}")
            print(f"Количество строк после очистки: {len(df_cleaned)}")
            
            # Перезаписываем исходный файл
            df_cleaned.to_csv(file_path, index=False)
            print(f"Файл успешно очищен и сохранен")
            
        except Exception as e:
            print(f"Ошибка при обработке файла {file_name}: {str(e)}")
    
    print("\nВсе файлы обработаны!")

if __name__ == "__main__":
    clean_csv_files()

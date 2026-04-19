import time
import os
import subprocess
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- НАСТРОЙКИ ---
PATH_TO_WATCH = "./"  # Директория для отслеживания
PYTHON_EXECUTABLE = sys.executable  # Используем текущий интерпретатор Python
EXTENSIONS_TO_WATCH = ['.py']  # Расширения файлов для отслеживания
# --- КОНЕЦ НАСТРОЕК ---

process = None
file_changed = False

class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global file_changed
        if event.is_directory:
            return
        
        watch_dir_abs = os.path.abspath(PATH_TO_WATCH)
        event_path_abs = os.path.abspath(event.src_path)

        if not event_path_abs.startswith(watch_dir_abs):
            return
            
        filename, extension = os.path.splitext(event.src_path)
        if extension.lower() in EXTENSIONS_TO_WATCH:
            print(f"Обнаружены изменения в: {event.src_path}")
            file_changed = True

    def on_created(self, event):
        global file_changed
        if event.is_directory:
            return

        watch_dir_abs = os.path.abspath(PATH_TO_WATCH)
        event_path_abs = os.path.abspath(event.src_path)
        if not event_path_abs.startswith(watch_dir_abs):
            return
            
        filename, extension = os.path.splitext(event.src_path)
        if extension.lower() in EXTENSIONS_TO_WATCH:
            print(f"Обнаружен новый файл: {event.src_path}")
            file_changed = True

def run_script():
    global process, file_changed
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "start.py")
    script_dir = os.path.dirname(script_path)

    print(f"Запуск скрипта: {script_path}")
    
    file_changed = False
    try:
        process = subprocess.Popen(
            [PYTHON_EXECUTABLE, "-u", script_path],
            cwd=script_dir
        )
        print(f"Скрипт запущен, PID: {process.pid}")
        return True
    except Exception as e:
        print(f"Ошибка запуска скрипта: {e}")
        return False

if __name__ == "__main__":
    watcher_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path_to_watch = os.path.abspath(os.path.join(watcher_dir, PATH_TO_WATCH))

    if not os.path.isdir(abs_path_to_watch):
        print(f"Ошибка: Директория '{abs_path_to_watch}' не существует")
        exit(1)

    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, abs_path_to_watch, recursive=True)
    observer.start()
    print(f"Наблюдатель запущен. Отслеживаю изменения в {abs_path_to_watch}...")

    try:
        if not run_script():
            print("Ошибка при первом запуске скрипта")
            exit(1)

        while True:
            if process and process.poll() is not None:
                exit_code = process.returncode
                process = None

                if exit_code != 0:
                    print(f"Скрипт завершился с кодом {exit_code}")
                    exit(1)
                else:
                    print("Скрипт успешно завершился. Перезапуск...")
                    time.sleep(1)
                    if not run_script():
                        print("Ошибка перезапуска скрипта")
                        exit(1)
            
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Наблюдатель остановлен пользователем (Ctrl+C)")
    finally:
        observer.stop()
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        observer.join()
        print("Наблюдатель завершил работу") 
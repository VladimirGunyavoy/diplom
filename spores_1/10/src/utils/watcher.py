import time
import os
import subprocess
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import select
import threading

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

class FileWatcher:
    def __init__(self):
        self.watched_file = None
        self.last_modified = None
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
    def set_file(self, filename):
        """Установить файл для отслеживания"""
        file_path = os.path.join(self.project_root, filename)
        if os.path.exists(file_path):
            self.watched_file = file_path
            self.last_modified = os.path.getmtime(file_path)
            print(f"Отслеживание файла: {filename}")
            return True
        else:
            print(f"Ошибка: Файл {filename} не найден")
            return False
            
    def check_file(self):
        """Проверить изменения в файле"""
        if not self.watched_file:
            return False
            
        try:
            current_modified = os.path.getmtime(self.watched_file)
            if current_modified != self.last_modified:
                self.last_modified = current_modified
                print(f"Файл изменен: {os.path.basename(self.watched_file)}")
                return True
        except FileNotFoundError:
            print(f"Ошибка: Файл {os.path.basename(self.watched_file)} больше не существует")
            self.watched_file = None
            self.last_modified = None
        return False

class ScriptRunner:
    def __init__(self):
        self.process = None
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.scripts_dir = os.path.join(self.project_root, 'scripts', 'run')
        self.running = True
        
    def run_script(self, script_name):
        """Запуск скрипта из папки scripts"""
        script_path = os.path.join(self.scripts_dir, script_name)
        
        if not os.path.exists(script_path):
            print(f"Ошибка: Скрипт {script_name} не найден в папке scripts")
            return False
            
        try:
            if self.process and self.process.poll() is None:
                print("Останавливаю предыдущий процесс...")
                self.process.terminate()
                self.process.wait(timeout=3)
            
            print(f"Запуск скрипта: {script_name}")
            self.process = subprocess.Popen(
                [sys.executable, "-u", script_path],
                cwd=self.scripts_dir
            )
            print(f"Скрипт запущен, PID: {self.process.pid}")
            return True
            
        except Exception as e:
            print(f"Ошибка запуска скрипта: {e}")
            return False
    
    def check_and_restart(self):
        """Проверка состояния процесса и перезапуск при необходимости"""
        if self.process and self.process.poll() is not None:
            exit_code = self.process.returncode
            if exit_code == 0:
                print(f"Скрипт успешно завершился, перезапуск...")
                time.sleep(1)  # Небольшая пауза перед перезапуском
                self.run_script(self.current_script)
            else:
                print(f"Скрипт завершился с ошибкой (код {exit_code})")
                self.running = False  # Останавливаем сторожа при ошибке
    
    def stop(self):
        """Остановить все процессы"""
        self.running = False
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()

def input_thread(runner):
    """Поток для чтения ввода пользователя"""
    while runner.running:
        try:
            script_name = input().strip()
            if script_name.lower() == 'q':
                runner.stop()
                break
            runner.set_script(script_name)
        except EOFError:
            break

def main():
    if len(sys.argv) < 2:
        print("Использование: python watcher.py <имя_скрипта>")
        print("\nДоступные скрипты:")
        scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts', 'run')
        scripts = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]
        for script in scripts:
            print(f"- {script}")
        sys.exit(1)

    script_name = sys.argv[1]
    runner = ScriptRunner()
    runner.current_script = script_name
    
    if not runner.run_script(script_name):
        sys.exit(1)
    
    try:
        while runner.running:
            runner.check_and_restart()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nПолучен сигнал прерывания, завершаем работу...")
    finally:
        runner.stop()
        print("Программа завершена")

if __name__ == "__main__":
    main() 
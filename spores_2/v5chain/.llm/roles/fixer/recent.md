# fixer — последние смены (новые сверху)
## 2026-10-12 yoga-v5chain-fixer-1 (ноут GunPC/WSL)
- Автоподъём после падения WSL: `~/.wsl_revive/{wsl_revive.py,agents.json,state.json,move_session.py,wsl_boot.sh}` (копии в `system/infra/machines/yoga/`), `/etc/wsl.conf [boot]`, crontab */2. Поднимает агентов реестра (`claude --resume`, свой cwd и CLAUDE_CONFIG_DIR!) + v8run, ремоунтит /mnt/c при 9p EIO. Пауза: файл `~/.wsl_revive/PAUSED`. Описание — `system/infra/machines/yoga/README.md`.
- pick_account на ноуте был пуст из-за жёсткого `/home/rl/` (теперь $HOME); call_fixer: `machines.json` `agent_name: laptop` (ключ машины остался yoga).
- Токены ноута не доходили до хаба: cron-синк смотрел в старый repo colab; починен, залито 2139 строк; statusLine в .claude-romaha/.claude-vladimirgun26 указывал на /home/rl.
- Дека: после переноса сессии между аккаунтами две копии транскрипта → дека читала старую; `move_session.py` убирает старую в `~/.claude-migrated-backup/`.
- Сторож токенов — по слову пользователя НЕ делаем (пульс/диспатчер); снят.

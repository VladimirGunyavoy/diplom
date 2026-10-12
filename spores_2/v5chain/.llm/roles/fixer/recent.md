# fixer — последние смены (новые сверху)
## 2026-10-12 laptop-v5chain-fixer-1 (ноут GunPC/WSL, romaha)
- WSL перезапущен пользователем 11:33, автоподъём сработал (research-3, worker-2/3, fixer, v8run). `wsl_boot.sh` теперь поднимает деку (tunnel+link) и почту (`watch-yoga` fixer-1/research-3) сам (после рестарта без терминала их не было; ждёт 15 с, чтобы .bashrc успел первым); при рестарте всё же вышли ДУБЛИ link/tunnel/почты yoga-ftl-worker-1 — убить не дали права, ждёт слова пользователя.
- Имена агентов ноута в графане: `yoga_usage_sync.py` пишет на хаб `sid_agent-<name>.txt` живых claude (из ~/.claude*/sessions). `hub_usage_sync.py` — в cron ноута.
- pick_account.py переписан по правилам A–G meta-1 (docstring + `.llm/roles/meta/proposal_limit_forecast.md`): возраст по cost, темп 30 мин, `--role`, 5ч по простою, 7д по запасу/стена 98, пробный запрос при устаревших данных (`claude -p ok --model haiku --output-format stream-json --verbose --no-session-persistence` отдаёт rate_limit_event со свежими %/resetsAt, ~$0.002). Тесты `infra/accounts/test_pick_account.py`.
- Грабля: `account_watchdog.py --help` НЕ показывает справку, а работает по-настоящему и шлёт алерт в tmux агентов.
- Пуш разрешён всем (слово пользователя 2026-10-12), доки поправлены; журналы `journals/` в gitignore.
## 2026-10-12 yoga-v5chain-fixer-1 (ноут GunPC/WSL)
- Автоподъём после падения WSL: `~/.wsl_revive/{wsl_revive.py,agents.json,state.json,move_session.py,wsl_boot.sh}` (копии в `system/infra/machines/yoga/`), `/etc/wsl.conf [boot]`, crontab */2. Поднимает агентов реестра (`claude --resume`, свой cwd и CLAUDE_CONFIG_DIR!) + v8run, ремоунтит /mnt/c при 9p EIO. Пауза: файл `~/.wsl_revive/PAUSED`. Описание — `system/infra/machines/yoga/README.md`.
- pick_account на ноуте был пуст из-за жёсткого `/home/rl/` (теперь $HOME); call_fixer: `machines.json` `agent_name: laptop` (ключ машины остался yoga).
- Токены ноута не доходили до хаба: cron-синк смотрел в старый repo colab; починен, залито 2139 строк; statusLine в .claude-romaha/.claude-vladimirgun26 указывал на /home/rl.
- Дека: после переноса сессии между аккаунтами две копии транскрипта → дека читала старую; `move_session.py` убирает старую в `~/.claude-migrated-backup/`.
- Сторож токенов — по слову пользователя НЕ делаем (пульс/диспатчер); снят.

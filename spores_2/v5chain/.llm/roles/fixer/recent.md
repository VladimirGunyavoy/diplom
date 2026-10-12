# fixer — последние смены (новые сверху)
## 2026-10-12 вечер laptop-v5chain-fixer-1 — аккаунты ноута
- Нумерация аккаунтов единая на хабе и ноуте (слово пользователя): 1 vladimirgun26 (ноут: `.claude`, он же `.claude-vladimirgun26` — дубль того же логина), 2 gun.vladimir26 (ноут: `.claude-gunvladimir26`, залогинен пользователем `claude auth login`), 3 romaha. Файл `~/claude-work/system/accounts/accounts.json` вне git; на хабе алиасы `machine_aliases.yoga`.
- Токен входа в git не хранится — на новой машине аккаунт надо залогинить руками. Новая папка аккаунта: скопировать settings.json (statusLine, autoMode), в `.claude.json` — hasCompletedOnboarding + trust проектов; `move_session.py <tmux> gun|romaha|default|<путь>`.
- pick_account: аккаунт без записей в логах опознаётся пробным запросом; выбор по max запас-в-час (romaha раньше сгорает) — для предпочтения gun/другого нужен явный флаг (не сделано).
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

## hub-v5chain-fixer-1 (2026-10-12)
Сделал: поднял meta-3; графана: имена агентов ноута (sid_agent-*, name в statusline), алиасы аккаунтов ноута, 7д без resets_at не рисуем + разрывы, скорость ctx ≥0, выбор дней недельного лимита (var-days в деке); hub_usage_sync.py (хаб→ноут) + open_log в pick_account; дека: realpath cwd и HOME (корень «нет statusLine на ноуте» — у процесса не было HOME); библиотека → system/library в рутинах; ответил про рестарт WSL. Урок: git на хабе без identity — GIT_AUTHOR_*/GIT_COMMITTER_* через env. Осталось: см. HANDOFF.

## hub-v5chain-fixer-5 (2026-10-05 … 2026-10-07)
Чинил/делал: aida после ребута (туннель сам); линии worker'ов/любой роли в machine/pulse/sentinel/pick_account/op_next_name; дека (любые файлы во вложениях, панель по проектам, перезапуск serve); launch_bg.py + баг send_verified (sid в проекте получателя, hop-); графана видит journal-2026 и ноутбук (agents_plot.load_all, feeder); пуш diplom по ssh. Ошибка: kill PID tmux-сервера (урок в infra.md). Осталось: коммиты в system не запушены. Смена сдана без преемника (режим Б).
## fixer-2 (2026-10-12 ~17:00, acc gun.vladimir26)
- A: pulse.py печатает «АВТОНОМНЫЙ РЕЖИМ…» при journals/_sentinel/AUTONOMOUS; account_watchdog — argparse (--help безопасен); pick_account.role_rate: конфиг → пересчёт из логов (медиана $/ч роли × %5ч/$ аккаунта) → таблица, + dispatcher/explainer (по логу сейчас worker≈12, research≈24 %/ч — проверить адекватность).
- B: yoga_usage_sync (fixer-1) уже в git; move_session сам добавляет неизвестные hop-* в agents.json. Дубль папки: .claude и .claude-vladimirgun26 — один email, но РАЗНЫЕ копии .credentials.json (риск рассинхрона refresh-токенов) — предложено пользователю оставить одну; не удалял.

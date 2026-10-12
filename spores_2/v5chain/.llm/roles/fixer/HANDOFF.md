# fixer HANDOFF — сдал hub-v5chain-fixer-1 (2026-10-12, по слову пользователя)
Открыто:
- **Реапер сессий на ноуте** (мало ОЗУ): написал laptop-v5chain-fixer-1 (14:xx→17:5x) — cron reap_idle.py */5 на ноуте, CHAIN_MAX_SESSIONS=6, touch finished_<имя> в конце смены агентов, wsl_revive не воскрешает finished. Ждёт его ответа (msg_from_yoga.log); при ответе «deck-<имя> не видит» — расширить reap_idle.py префиксы.
- **Дубли деки на ноуте** (link/tunnel/почта yoga-ftl-worker-1) висят, убить — только по слову пользователя.
- **Графана:** алиасы ноута — accounts.json machine_aliases.yoga ({.claude→.claude-vladimirgun26, .claude-gunvladimir26→.claude}); проверить, что расход laptop-v5chain-fixer-2 (gun.vladimir26) попал в серию acc2. Имена агентов ноута — sid_agent-<имя>.txt, их пишет yoga_usage_sync.py на ноуте.
- feeder.py/mkdash.py (~/grafana, вне git): параметр days, разрывы 7д-линий >3ч, переменная days (дека передаёт var-days). feeder — ручной процесс (python3 feeder.py в ~/grafana), после правок agents_plot.py перезапускать.
- Всё закоммичено и запушено в claude-system (identity: GIT_AUTHOR_/COMMITTER_NAME/EMAIL через env, в git config хаба её нет; пользователь разрешил коммитить/пушить 2026-10-12).

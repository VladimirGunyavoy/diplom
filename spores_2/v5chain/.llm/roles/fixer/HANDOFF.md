# fixer HANDOFF — сдал hub-v5chain-fixer-1 (2026-10-12, по слову пользователя)
Открыто:
- **Реапер сессий на ноуте** (мало ОЗУ): написал laptop-v5chain-fixer-1 (14:xx→17:5x) — cron reap_idle.py */5 на ноуте, CHAIN_MAX_SESSIONS=6, touch finished_<имя> в конце смены агентов, wsl_revive не воскрешает finished. Ждёт его ответа (msg_from_yoga.log); при ответе «deck-<имя> не видит» — расширить reap_idle.py префиксы.
- **Дубли деки на ноуте** (link/tunnel/почта yoga-ftl-worker-1) висят, убить — только по слову пользователя.
- **Графана:** алиасы ноута — accounts.json machine_aliases.yoga ({.claude→.claude-vladimirgun26, .claude-gunvladimir26→.claude}); проверить, что расход laptop-v5chain-fixer-2 (gun.vladimir26) попал в серию acc2. Имена агентов ноута — sid_agent-<имя>.txt, их пишет yoga_usage_sync.py на ноуте.
- feeder.py/mkdash.py (~/grafana, вне git): параметр days, разрывы 7д-линий >3ч, переменная days (дека передаёт var-days). feeder — ручной процесс (python3 feeder.py в ~/grafana), после правок agents_plot.py перезапускать.
- Всё закоммичено и запушено в claude-system (identity: GIT_AUTHOR_/COMMITTER_NAME/EMAIL через env, в git config хаба её нет; пользователь разрешил коммитить/пушить 2026-10-12).
# HANDOFF fixer (2026-10-12 ~15:15, от laptop-v5chain-fixer-1)
ТЕКУЩАЯ ЗАДАЧА — в `journals/_sentinel/task_laptop-v5chain-fixer-2.txt` (учёт acc2/vladimirgun26: две папки одного аккаунта, wsl_revive для worker-5/новых hop-*, дубль папки). Уже сделано: yoga_usage_sync синкает все локальные usage_log проекта (запушено), pick_account по resets_at (5628119).
Графана-баги пользователя (7д-график, отрицательная скорость контекста) хабовый fixer починил (agents_plot.py).
Старое открытое:
1. **Графана (владелец — хабовый fixer, `~/grafana/feeder.py` на хабе; ему отправлено):** (а) «Недельный лимит (7д)»: между 10-08 и 10-12 нет точек, линия идёт к 0 — в серии есть фальшивые нули/1% (hub-лог: сотни записей romaha/.claude с seven_day_pct 0–1 и five_hour 0–8 в 10-12 00:00 и 07:30–09:30 — похоже, statusline пишет 0 при rate_limits=null); acc2 прыгает 96→1→96 (11:29, 12:22→14:21). Нужно отбрасывать записи с нулём без resets_at / не рисовать через пропуск. (б) «Скорость роста контекста»: отрицательный рост (оранжевая пунктирная до −10 %/ч около 12:45–13:30) — контекст падает после compact/нового сессии; клипать ≥0 или помечать сброс.
2. **Дубли деки после рестарта 11:33**: два link/tunnel/watch-yoga yoga-ftl-worker-1 (пары процессов ppid=1 и ppid=15). Нужно слово пользователя убить лишние. wsl_boot.sh исправлен (на следующий рестарт), не проверен рестартом.
3. **pick_account**: не сделано — пересчёт role_rate из лога; в рутинах (spawn-шаблоны worker/research/searcher/dispatcher/explainer) добавить `--role <роль>` — дело meta-1 по DOC_CHANGES. Ждёт повторной проверки на живых вызовах.
4. Реестр ревайва: research-3, worker-2/3, hop-laptop-v5chain-fixer-1 (мой sid 69e1a3fb); новый fixer — заменить запись (sid, config_dir).
5. Старые пункты: usage хаба в графане ноутбука (проверено — имена laptop-* видны), падения /mnt/c (9p EIO) — причина не найдена.
Пульс выключен до задачи.

# fixer — последние смены (новые сверху)

## hub-v5chain-fixer-1 (2026-10-12)
Сделал: поднял meta-3; графана: имена агентов ноута (sid_agent-*, name в statusline), алиасы аккаунтов ноута, 7д без resets_at не рисуем + разрывы, скорость ctx ≥0, выбор дней недельного лимита (var-days в деке); hub_usage_sync.py (хаб→ноут) + open_log в pick_account; дека: realpath cwd и HOME (корень «нет statusLine на ноуте» — у процесса не было HOME); библиотека → system/library в рутинах; ответил про рестарт WSL. Урок: git на хабе без identity — GIT_AUTHOR_*/GIT_COMMITTER_* через env. Осталось: см. HANDOFF.

## hub-v5chain-fixer-5 (2026-10-05 … 2026-10-07)
Чинил/делал: aida после ребута (туннель сам); линии worker'ов/любой роли в machine/pulse/sentinel/pick_account/op_next_name; дека (любые файлы во вложениях, панель по проектам, перезапуск serve); launch_bg.py + баг send_verified (sid в проекте получателя, hop-); графана видит journal-2026 и ноутбук (agents_plot.load_all, feeder); пуш diplom по ssh. Ошибка: kill PID tmux-сервера (урок в infra.md). Осталось: коммиты в system не запушены. Смена сдана без преемника (режим Б).

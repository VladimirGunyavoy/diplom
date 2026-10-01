# links_archive
- сборщик v5chain (acc2, Opus, 2026-09-29): собрал v5chain = код v4_ort + `.llm`; цель записана в TASK/PLAN. Грабли: скрипты system/infra берут проект из cwd; `src/atlas/` — отдельный пакет, Ursina не импортировать; сбой классификатора «no verdict» — временный, повторить.

- hub-worker-4 (acc2, Sonnet, 2026-09-30): v6 дифдрайв `dd_atlas.py` (ромб/RECT, агент, коридор SLSQP, диски), манипулятор `manip2.py` (V==Дейкстра), H1 DI `adaptive_di.py` (N≈117 V/T*=1.02 vs решётка N=2401 1.30). Грабли: cron/sid/журнал только в v5chain; pgrep -f в until находит сам себя; sleep>120 блокируется; коммит -c user.name/-c user.email, git add только своих путей; rm -r блокируется.

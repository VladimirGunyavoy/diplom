# RESEARCH_HANDOFF — hub-research-1 (живой, заменять)
## Сделано (2026-09-29)
- Обсуждение с пользователем → `spores_2/v6/docs/source_doc.md` (SPORE v6); TASK/PLAN п.0 под v6; worker hub-worker-3 уведомлён (send_verified УСПЕХ).
- Коридор ctx research 40/50/60 (pulse.py, RESEARCH_ROUTINE); правило «спорить с пользователем» — в роли.
- aida: связь есть; сторож туннеля `~/tunnel/aida_tunnel_guard.sh` (cron @reboot + */5), infra.md.
- Тест DI: позднее переключение на δ → потеря ≈4√δ (0.05→0.86, 0.1→1.28, 0.2→1.96).
## Очередь research (пока пользователя нет)
1. Тест: плотность точек по гессиану vs равномерно (DI, одинаковое число точек), ошибка V — `reports/research/`, заметка `knowledge/research/`.
2. Проверить ссылки source_doc [проверить]: Kaya & Noakes (switching-time optimization), Alauzet/Loseille (hessian-metric adaptation) — WebSearch; снять пометки.
3. Прототип-эталон для worker: клетка DI + V в 2n+1 точках + переключение по знаку ∂V/∂v — воспроизводит ли T* (малый numpy, `reports/research/`).
4. ρ_max/r/τ для DI в нормированных координатах (x/4, v/2) — числа «на бумаге» для worker.
5. Отвечать worker на вопросы по идее; сверять его шаги v6 с source_doc.

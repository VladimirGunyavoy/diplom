# links_recent — последние 5 звеньев (новые сверху)

## hub-worker-12 (acc1, Sonnet, 00:08–10:15), причина смены: ctx 28% (софт 30%)
Сделано: маятник на faces: привязка проб к линиям + `bigfin` (по диагнозу research-7) → 100% дошли, T/Ta 1.001 (v6 PendAtlas); затем по рекомендации research-7 написан `v7/src/cells7/spore_v.py` (V по клеткам cover_layer+гало, V*=min_k[Δt+V*(φ_k)], адаптивные узлы, `fill_gaps`, периодичность): DI 100%, T/T* 1.15 (5 перекл.), маятник 98%, T/Ta med 1.32 (18). Отчёт `v7/reports/spore_v_di.md`, скрипты+кэши `v7/reports/spore_v/` (в скриптах пути /tmp/claude-1000 — подправить).
Стоп на / следующий шаг: (1) **идёт** `/tmp/claude-1000/s8.py` → `s8.log` (DI с `blend=True` — взвеш. среднее вместо min; гипотеза: min по клеткам смещает V вниз, V*<T* у 97% узлов, V/T* med .87); если жив — забрать итог (V*/T* med, T/T*, dt .06/.02), иначе перезапустить (~40+ мин, bincount); (2) LQR-клетка у цели маятника (`hold.py`, sat(−Kx), xᵀPx≤.5, research-7: скачки у цели 15→0) в агент spore_v; (3) спектр u (11 значений) / L1-штраф (v7_spectrum_cost.md): в Беллмане и агенте непрерывный набор; (4) ответ hub-research-7 на сверку spore_v vs v7_spore_di.py (отправил список расхождений; его гипотеза (а) шаг агента=шагу Беллмана НЕ подтвердилась: dt .06→1.13, .02→1.49, .0075→2.68); (5) адаптивность граней (faces) — отложена рекомендацией research-7.
Грабли: `pkill -f` убивает шелл — убивать по pid; `np.add.at` в Якоби катастрофически медленный (использовать bincount); cover_layer m=40 ≈ 1000 с на слой — не пересчитывать, брать pkl; 5% точек вне всех клеток (дыры) → V=BIG, агент встаёт: всегда `fill_gaps`; агент маятника накручивает обороты — `S.wrap` в `_pairs` (иначе BIG); гистерезис eps=.003 не помог (DI, маятник); git index.lock бывает от чужих агентов — повторить; тяжёлое только через `systemd-run --user --scope -p MemoryMax`, в фоне `setsid nohup … & disown`; фоновые прогоны без setsid умирают молча.
Решения цепочки: гипотеза «физический предел» маятника отозвана (слово пользователя); faces→spore_v по рекомендации research-7 (v7_spore_cells.md).
Коммиты: v7: cb67026, ce614a2, 2eb84ca…; цепочка: PLAN/STATUS обновлены (не пушил: STATUS «не пушить»).
Токены: T_START 5ч 17% / ctx 4% / $0.10 (00:08) / T_BEFORE_END 5ч 25% / ctx 27% / $10.1 (10:15)
NEXT_LINK: hub-worker-13   NEXT_MODEL: sonnet

## hub-worker-11 (acc3, sonnet, 20:10–00:40), смена: ctx 28%
v7: градиент V (value_jet.py) на DI — не сошёлся (v7_cells.md); hold.py/holdn.py спектр+LQR (DI/маятник/манип. 2 зв.; v7_hold.md); данные для research-7 (exits_summary.md); **faces.py — общие грани по research-7 (v7_faces.md): DI 1.025 при 82 линиях, маятник — достижимость падает на мелких сетках (открыто)**. Грабли: pkill -f убивает шелл; тяжёлое — systemd-run MemoryMax=3G; sleep>120 блокируется (until-цикл); git -c user.email/name обязателен; не пушить.

## hub-worker-10 (acc3, sonnet, 16:44–21:40), смена: human (NEEDS-HUMAN п.3)
v7: locate предфильтр ×2, симметричная проверка ядер, nD-клетка celln.py, покрытие 9 систем (v7_summary.md), п.5 эллипс+пул (ellipse.py), картинки dd/m2g. V по клеткам (граф/решётка/гало) не работает — v7_cells.md. Грабли: v6 импортируется по именам файлов; nD-метрики O(N²), 8D≈20 мин; sleep>120 блокируется; git -c user.email обязателен; не пушить. Код v7 на aida ~/spore_v5/w10/v7.

## hub-worker-9 (acc3, sonnet, 09:20–16:50), смена по ctx 21%
`corridor_query(fs,neigh,wlim)`, refine |w|≤WM; v6_summary (WM3/6, neigh8, dd+диски 40/40, маятник, двойной маятник g=1/1.5/2); v7 каркас `spores_2/v7/src/cells7` (systems, cell, cover) + tests/run_cover.py. Грабли: ssh -f + setsid nohup; sleep>120 блокируется; casadi нет; pkill -f убивает шелл; git add только своих путей. Код v7 на aida `~/spore_v5/w9/v7`.

## hub-worker-8 (acc2, sonnet, 02:40–09:30), смена по ctx 22%
ночные эксперименты (а)–(д): статистика 3зв/2зв/dd/маятник, 8D 100k 8/8, sweep, адаптив vs сетка, `v6/reports/v6_summary.md`; `back_heuristic` WH=1.5 при препятствиях. Грабли: aida `~/spore_v5/w7/v6` (scp -r src tests); `OMP_NUM_THREADS=1 nohup`, фоновый until; env тестов G,DOWN,UP,DT,WH,KN,WM,NQ,NAME,OBST.



# Session Handoff — линия B (переоткрыта research-25: профилирование/ускорение спор)
[РУТИНА НЕ ВЫПОЛНЕНА]
Звено: hub-v5chain-worker-b9 (acc2, sonnet, 22:52–~01:45), причина смены: ctx ~28% (софт 30%) + 5ч-лимит ~68%
Сделано (всё закоммичено, отчёты `v7/reports/speed/`): SPD1 retarget manip 361→11 с (spd.py, кэш рёбер, tol/lcut/fp32); SPD1b сетка по q2 без досева ≈14 с (досев 205 с — не годится); FIN-ω (`FINBOX`, 3/20 нарушали |ω|; manip 1.062/1.108/1.502);
FORB (`growN.GF`/`free`, `FORB=1 FD TY ML`; 0 нарушений, 9/13; 4 застрявших разбирает c3); SPD2: manip запрос 520→210 мс (HSF .1), профили дд (финиш 94%) и маятника (поиск клетки 77%); SPD4: компакт-рёбра GPU 3.96→0.91 ГБ (retarget +40%).
Стоп на / следующий шаг (порядок research-25: SPD2 → SPD4 → SPD3 → RACE-споры последними):
1. SPD2 дд: в `grow3.rollout` заменить `shoot` (SLSQP по всем топологиям, 683 мс/старт) на `finish_gen.shoot_pol`; проверить 60 стартов T/refbox ≈1.004 (aida `~/spore_v5/b9/profq.py grow3`, команда в spd2.md; профиль-обвязка там же).
2. SPD2 маятник: ускорить `_test` (поиск клетки 77% запроса, 17–37 мс): старт поиска с клетки прошлого шага / GPU.
3. SPD4 остаток: `save_compact` на диск (кэш рёбер 4 ГБ → ~0.9), ОЗУ solve (списки I_/IDX_/W_ + argsort), qx (X 512 Б/узел float64), `HexIdx.add` 122 с. Число клеток НЕ трогать (слово пользователя).
4. SPD3: построение слоёв manip (270–343 с): профиль cProfile `Atlas()` (БУДЕТ HexIdx.add/_key/growN), правки по профилю.
5. RACE-споры (последними): `knowledge/research/race_track.md`, PLAN RACE/RACE-5.
Грабли: aida GPU общий (`flock /tmp/gpu.lock`, часто занят c-линией); `pkill -f` по шаблону убивает свой ssh — писать `[x]xx`-шаблон; Pool(fork) нельзя после инициализации CUDA (spd.grow_layers звать до GoalSolver); рабочая папка aida `~/spore_v5/b9/` (L_g300so.pkl, V_base.npy, E_g300so_*, A_forb1.pkl, скрипты exp*.py/prof*.py/finspeed*.py, ocp3.npy — OCP-эталоны для q2g); qx для rollout строится ~125 с при первом вызове — в таймер не включать (ошибка «2.4 с» в forb.md исправлена); git pull конфликтует в JOURNAL.md (оставлять обе стороны); общий working tree с c-линиями — `git checkout` не использовать.
Решения цепочки (автономный режим): FINBOX по умолчанию вкл. для manip (честный финиш); досев цели не штатный; HSF .1 в доводке финиша (ΔT 3e-5).
Токены: T_START 5ч 16% / ctx 5% (22:52) / T_BEFORE_END 5ч ~68% / ctx ~28%
NEXT_LINK: hub-v5chain-worker-b10   NEXT_MODEL: sonnet

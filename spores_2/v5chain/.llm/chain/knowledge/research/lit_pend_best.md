# lit_pend_best — лучшая прошлая версия спор на маятнике и её фичи (searcher-5 для research-6, 2026-10-13)
Только факты из отчётов/кода, без рекомендаций. Читал: `versions_compare.md`, `lit_v8_regress.md`, `lit_pend_insights.md`, `cut_walls.md`, `nf_curved_ends.md`, `v_kink_q.md`; код v7 `grow_cells2d.py` (оглавление, grep флагов), v8 `growN.py`/`solve8.py`/`main_pend.py` (grep). Чисел заново не мерил; [?] — не проверено.
## (1) Лучшая версия и числа
**v7 `v7/src/cells7/grow_cells2d.py`** (SYS=pend, 2D; перенос w20 из r14–r19; `experiments/pend/compute.py`). Постановка всех цифр v7: φ̈=sinφ+u, |u|≤UM=.3, слои −UM/0/+UM нет — **два/три атласа по u** [?: pend у v7 по отчётам «двухслойный ±u», `lit_pend_insights.md` шапка], цель — ящик |φ|,|ω|≤RHO .1 (`:11,:31`), φ периодична 2π, поле ω ±WL (WS 3.5), DTN .06, RMAX .3, M=5, TMAX 3; эталон `reports/research/pend_ref_T.npy` (100 стартов rng 0, стрельба ≤3 дуг), 5 зёрен, T/эт = T агента/эталон, дошли 100%.
| вариант (v7 pend, u .3) | T/эт ср. | max | спор | узлов | время | источник |
|---|---|---|---|---|---|---|
| **NORMFRONT=2, DELTA .03, без стен** (лучший T) | **1.0085** | 1.16–1.19 | 686 | 120k | атлас/solve не указаны | `nf_curved_ends.md` п.4 |
| «жирные» кольца + **MADAPT .001** + BFINE 8, LOOK 20 (лучший по спорам при T≈1.009) | **1.0089** | 1.26 | **237** | 78k | атлас 80 с, solve 40 с | там же п.9 |
| рекомендованная связка 252: NF+OWN+CUTR+LOOK20 | 1.015 | — | ~895 | — | 193–225 с | `lit_pend_insights.md`, `cut_walls.md` |
| OWN+CUTR+LOOK20 (стены) | 1.0128 | — | 1010–1191 | — | — | `lit_pend_insights.md` |
| быстрый (серия 84) | 1.040 | — | — | 37k | атлас 38 с, solve 7 с, запрос 0.09 с | `lit_pend_insights.md` |
| конкуренты: CUTR без LOOK (GROW2+GOALB+CUTR) | 1.028 | — | 1190–1290 | — | 170–240 с | `cut_walls.md` таблица |
| без стен и NF (GROW2+GOALB, CUT) | 1.023 / без стен 1.050 | — | 1550 / 517 | 68–74k | ~60 с | `cut_walls.md` |
p90 в отчётах v7 по маятнику не приводится [?]; только max по зёрнам. LOOK 20 даёт запрос ≈7.7 с/старт (жирные 239 кл.), LOOK 0 — 1.062 / 0.14 с (`nf_curved_ends.md` п.10).
Ближайшие конкуренты: (а) v7 CUTR-связка (1.015–1.028 при ~900–1300 спор); (б) v6 `atlas6` (V по сетке спор + Q-жадный агент, «6/6 до верха» без числа T/эт: коммит 8db9f9f; v6 H1 на ДИ 1.022/117 спор — другая система).
v8 pend сейчас (b3, `v_kink_q.md`): commit мед **1.029/p90 1.169/max 1.45**, look 1.021/1.148/1.50 (v7 LOOK20 1.016/1.032), рост 8–13 с, solve 11–23 с, 346 кл.; цель — круг RHO .1 (v7 — ящик), слои −UM/0/UM, DTN .06 в main_pend, эталон тот же `pend_ref_T`.
## (2) Фичи лучшей версии (v7 `grow_cells2d.py`, если не указано иное)
- Клетка = спора + нормальный отрезок, поток rk4 вперёд/назад ≤TMAX: `Cell.build :37`, `normal :33`, `step :30`.
- Рост grow2: ящик [klo,khi]×[ilo,ihi], 1 шаг в 4 стороны; стороны гаснут по полю/стене, изгибу, покрытию, самонаезду, SMAX: `_grow2_gen :264`, `bend :287`, `ok_rect :293`, `endcov :298`.
- **Фронт NORMFRONT=2**: узел клона j = пересечение его траектории с прямой ⟂ потоку через узел соседа: `nf_rows :193`, флаг `:184`; изгиб DELTA (метрика грамиана `NORM=gram`) — ручка «T ↔ узлы» (`DELTA :16` умолч. .03).
- Гало HALO .1, нахлёст OVL .05 (`:12`), покрытие соседями FRAC/OVH; OWN — стоп торца по покрытию СВОИХ столбцов (`:182,:297–302`).
- SMAX/SLAM — стоп по расширению фронта w/w0 (`:358–361`); SELFOV — кольца без самонаезда (`:349`, TMAX 8); NFDEAD/BFINE — мёртвые узлы и мелкая цепочка (`:44–48`).
- MADAPT 5→9→17→33 и MROW подклетки (`:375–389`, `split_rows :423`, `trim_rows :449`).
- Посев: `build_all :401`, `build_layer :542`, SEEDEPS (`:409,:550`), GOALB — клетки от цели (compute.py).
- Периодика θ: `wrap :25`, `SH=(0,PER,−PER) :34`; бины QIdx/Index с копиями `:104–109,:162–165`; locate `QIdx.query :117`, `_test :140`; `inbox` по θ не ограничивает (`:32`). Подробности — `lit_clones.md`.
- Стены CUT/CUTR: `BARRIER/nearb :180–181`, `BTOUCH`, `BEPS .01`; `compute.py:26–36,94` (проход 0 → бисекция 6× по метке траектории агента → KD-дерево стен → проход 1).
- Расчёт V: `Atlas.solve :682` — V(узел)=min_u[DTN+V*(φ_u(узел,DTN))] (шаг ПОСЛЕ Δt, не 0-временной), V* = min по всем перекрывающим ячейкам билинейно (`vstar :674`, `stencils :629`, `interp :666`), `tgoal :678` — точный вход в ящик цели, JUMP/JMODE — разрыв V (`:14`).
- Агент: `rollout :745` — шаг DTN, argmin по управлениям; `look :727` — прокатка L шагов (LOOK 20 = 1.2 с).
- Вне файла: exact_switch (σ-нуль, `reports/research/exact_switch_pend.py`), финиш `finish_gen.shoot` (VF), LQR-клетка цели (`spore_v.rollout(hold,hc) :92`) — в рекомендованную связку 252 не входили [?].
## (3) Есть ли в v8 сейчас
| фича | v8 (growN `v8/src/algo/growN.py`, solve8) | отличие |
|---|---|---|
| клетка/поток rk4 | `Cell :171`, `step :82`, `adv :83` | nD; M=5 в main_pend |
| NORMFRONT=2 | `nf2_rows :90`, флаг `:70` (pend умолч. 0; b3 ставит 2) | реализовано; DELTA в main_pend .5/.2 против v7 .03 |
| изгиб DELTA, грамиан | `bend :388`, `NORM=gram` | есть; другое значение |
| гало/нахлёст | HALO .1, OVL, FRAC .5, OVH .5 (main_pend) + ZACEP/ZOVMAX/ZFRAC/ZSEED (`:141`), `halt` | зацеп в v8 новый; OWN нет |
| SMAX | `:19`, стоп `:434,:486` (только m_==1, умолч. 0) | есть, выкл. |
| TAUMIN/fold (схождение фронтов) | `:141,:435–440` | только в v8 |
| SELFOV, OWN, NFDEAD, BFINE, SEEDEPS, MADAPT, MROW | grep: 0 вхождений | нет |
| посев | `build_layer :522`, `covcheck/covdone :528–541`, COVTOL .003, `goal_seeds :514` | посев в непокрытое по доле (v7: SEEDEPS + GOALB) |
| периодика θ | `PERIOD`, `SH :72–73`, `wrapy :74`, `HexIdx` с копиями `:229–236` | nD-вариант того же приёма |
| стены CUT/CUTR | не используются: заменены ограничением роста ширины фронта (слово пользователя через research-6; SMAX/`stretch` `growN.py:434,:486`, TAUMIN) | `BARRIER/nearb :166` остался в коде, 2-проходности нет |
| расчёт V | `solve8.Graph.solve :226`, `_solve_q :313`, E1/E2/E3, `_bisect_pass :258`, XENTRY | граф рёбер по узлам; переход слой→слой в узле, не «шаг Δt в любую точку» (`lit_v8_regress.md` (а)) |
| V* чужого слоя | `Vlayers :354`, `Vstar :381`, PESS | v7: min по ячейкам; v8: Q-опции, PESS |
| агент | `run_commit :401`, `run_agent :428` dt .01, eps 1e-3; look = 1 шаг Беллмана | v7 DTN .06, LOOK 20 |
| exact_switch / финиш | `finish_gen.py` есть, не вызывается; FINGRID 1 вхожд. в growN | подключено частично (commit ≈ упрощённый exact_switch, `versions_compare.md`) |
| LQR-клетка цели | нет | — |
| цель | круг (`GOALSHAPE='ball'`), RHO .1 | v7 ящик |
| слои | −UM, 0, +UM (`ULAYERS`) | v7 pend — ±u |
## Не нашлось
p90 по маятнику у v7; время атласа/solve для NF2 DELTA .03; прямого прогона v7-лучшего на круге/слое u=0; числа v6 по маятнику (T/эт).

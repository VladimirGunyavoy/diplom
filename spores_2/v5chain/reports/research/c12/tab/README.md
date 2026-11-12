# c12/tab — подготовка к таблице «где атлас лучше графа» (06:45, research-39). Код БЕЗ запусков.
- `tvg_vt.py` — среда: ДИ 4D/6D/8D (`_mk_dix`, 81 управление в 8D), `BOX=full` для manip/dp1; `tvg_dl.py` — `run_agent` теперь считает **нарушения** (common.verify): `viol_box` (выход за коробку), `viol_nogoal` (не дошёл за tmax=40), `viol_slow` (дошёл, но > 3·T_ref); `tvg_lp.py` — r37 с LP-порциями.
- `dix_ref_exact.py NK SEED NH` — эталон ДИ (NH=3 → 6D, 4 → 8D), как `di6_ref_exact.py`; `run_row.sh` — одна строка (параметры по среде, time -v).
- Атлас: `growNa.py` (v7/reports/big/10_adapt) знает SYS = dd, pend, manip, di4, dp1, m3d, race, race4 — **ДИ 6D/8D в growNa нет**; атлас есть для 2D (dd), 4D (di4, manip, dp1), 6D (m3d). ДИ6D/8D-строку атласа сделать нельзя без нового SYS (отдельное решение).
- fb: `../fb/fb.py`. Память: `../../sampler.sh` (на aida2 `~/spore_v5/c12/sampler.sh`).

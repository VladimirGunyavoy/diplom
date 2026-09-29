# 🔴 Agent End Routine

**Last updated:** 2026-09-29 (v5chain: замеры токенов из лога)

> Задача финишной рутины — **только handoff**. Всё остальное (changelog, decisions,
> обновление state/, коммит рабочих файлов) сделает следующий агент в начале своей
> сессии — см. `AGENT_START_ROUTINE.md`.

---

## 1️⃣ Замер токенов 3: до финишной рутины (из лога)

```bash
python3 ~/claude-work/system/infra/tokens/last_usage.py | grep -E '"(five_hour_pct|ctx_pct|cost_usd|seven_day_pct)"'; date "+%F %T"
```

Запомни как **T_BEFORE_END** (`5ч N% / ctx M%`, `$`).

---

## 2️⃣ Напиши state/session_handoff.md

```markdown
# Session Handoff
[РУТИНА НЕ ВЫПОЛНЕНА]

Дата: {ДАТА}, сессия N
Сделано: {одна строка}
Стоп на: {где остановились / что не закончено}
Следующий шаг: {конкретное действие}
Грабли: {что неочевидно, на что наступили}
Не трогай: {что хрупко или требует осторожности}
Рабочие файлы: {список файлов изменённых в этой сессии — для git add следующим агентом}
Архитектурные решения: {кратко если были, или "нет"}
Коммиты этой сессии: {git log --oneline -3}
```

> Пометка `[РУТИНА НЕ ВЫПОЛНЕНА]` обязательна — новый агент проверяет её наличие.

---

## 3️⃣ Git коммит handoff

```bash
git add spores_2/v5chain/.llm/state/session_handoff.md
git commit -m "$(cat <<'EOF'
[v5chain sN]: session handoff

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## 4️⃣ Замер токенов 4: после финишной рутины (из лога)

```bash
python3 ~/claude-work/system/infra/tokens/last_usage.py | grep -E '"(five_hour_pct|ctx_pct|cost_usd|seven_day_pct)"'; date "+%F %T"
```

Запомни как **T_END** (`5ч N% / ctx M%`, `$`).

---

## 5️⃣ Сводка пользователю

```markdown
## Сводка сессии N
✅ Сделано: ...
📝 Рабочие файлы: ...
🔜 Следующие шаги: ...
📊 Токены: Старт={T_START}% | После рутины={T_AFTER}% | До финиша={T_BEFORE_END}% | Конец={T_END}%
   Работа: {T_BEFORE_END - T_AFTER}% | Финишная рутина: {T_END - T_BEFORE_END}%
⚠️  Стартовую рутину сделает следующий агент (changelog, decisions, коммит рабочих файлов)
```

---

## ⚠️ Что НЕ делать в финишной рутине

- ❌ Не обновлять `history/changelog_recent.md` — это задача следующего агента
- ❌ Не обновлять `state/current.md`, `state/plan.md`, `state/issues.md`
- ❌ Не коммитить рабочие файлы — только `session_handoff.md`
- ❌ Не обновлять `state/token_stats.md`

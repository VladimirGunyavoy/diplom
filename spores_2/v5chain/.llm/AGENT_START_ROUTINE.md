# 🟢 Agent Start Routine

**Last updated:** 2026-09-29 (v5chain: замеры токенов из лога)

> Выполняй это **В НАЧАЛЕ** каждой сессии, ДО начала работы над задачей.
> Контекст сейчас почти пустой — это лучший момент для рутины: тяжёлая часть
> (changelog, decisions, обновление state/, git commit) больше не конкурирует
> с накопленным за сессию диалогом.

---

## 0️⃣ Замер токенов 1: до рутины (из лога)

⚠️ **ЭТО ПЕРВОЕ ДЕЙСТВИЕ СЕССИИ.**

Одной командой, ДО чтения handoff и state/:
```bash
python3 ~/claude-work/system/infra/tokens/last_usage.py | grep -E '"(five_hour_pct|ctx_pct|cost_usd|seven_day_pct)"'; date "+%F %T"
```

Запомни как **T_START** (`5ч N% / ctx M%`, `$`, `7д%`, время). Лога нет — спроси пользователя (см. `AGENT_START.md`).

---

## 1️⃣ Прочитай handoff

Открой `state/session_handoff.md`.

Убедись, что в начале файла стоит пометка `[РУТИНА НЕ ВЫПОЛНЕНА]` — это значит,
что предыдущий агент корректно завершил сессию (написал только handoff и закоммитил его).

---

## 2️⃣ Обнови state/

- **`state/current.md`** — обнови «что работает / в процессе / сломано» на основе handoff
- **`state/issues.md`** — пометь решённые (`~~зачеркнуть~~`), добавь новые из «Грабли»
- **`state/plan.md`** — вычеркни выполненное, добавь следующие шаги из handoff

---

## 3️⃣ Задокументируй в history/

Добавь запись о **предыдущей** сессии в `history/changelog_recent.md` (вверх, после заголовка):

```markdown
## YYYY-MM-DD (vX сессия N / сессия M) - Краткая тема

**Что сделано:**
- 🆕/🔄/🐛/🗑️ `файл` — что изменилось

**Технические детали:**
- ...

**Участники:** Пользователь + Claude X
```

Если в `changelog_recent.md` уже 5 записей — перенеси самую старую в конец `history/changelog_archive.md`, затем добавляй новую.

Если в handoff упомянуты архитектурные решения — запиши их в `history/decisions.md`.

---

## 4️⃣ Git коммит

```bash
git status          # посмотри что изменилось
git diff            # проверь изменения
```

Добавляй файлы явно — рабочие файлы из поля «Рабочие файлы» в handoff + файлы рутины:

```bash
git add {рабочие файлы из handoff}
git add spores_2/v5chain/.llm/state/current.md
git add spores_2/v5chain/.llm/state/plan.md
git add spores_2/v5chain/.llm/state/issues.md
git add spores_2/v5chain/.llm/history/changelog_recent.md
# (если была ротация) git add spores_2/v5chain/.llm/history/changelog_archive.md
# (если были решения) git add spores_2/v5chain/.llm/history/decisions.md

git commit -m "$(cat <<'EOF'
[v5chain sN]: краткое описание сессии

- что сделано
- технические детали

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

❌ Никогда `git add .` или `git add -A`
❌ Не пушить — только если пользователь явно попросил

---

## 5️⃣ Замер токенов 2: после рутины (из лога)

```bash
python3 ~/claude-work/system/infra/tokens/last_usage.py | grep -E '"(five_hour_pct|ctx_pct|cost_usd|seven_day_pct)"'; date "+%F %T"
```

Запомни как **T_AFTER** (`5ч N% / ctx M%`, `$`).

---

## 6️⃣ Обнови token_stats.md

Добавь новую строку с T_START и T_AFTER (T_BEFORE_END и T_END заполнит финишная рутина в конце этой же сессии):

```
| {N} | {ДАТА} | {T_START}% | {T_AFTER}% | — | — | {T_AFTER-T_START}% | — | — |
```

Закоммить отдельно:

```bash
git add spores_2/v5chain/.llm/state/token_stats.md
git commit -m "$(cat <<'EOF'
[v5chain sN]: стартовая рутина и token stats

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## 7️⃣ Приступи к задаче

Теперь можно работать. Если пользователь уже написал задачу в первом сообщении — берись за неё сейчас.

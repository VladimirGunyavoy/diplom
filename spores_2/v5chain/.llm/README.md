# LLM Context Directory — spore / v5chain

**Last updated:** 2026-09-29 (создание v5chain: цепочка агентов по ролям поверх контекста v4_ort)

## 🎯 Главная цель
Минимизировать токены при передаче контекста между AI-сессиями: агент читает только то, что нужно для задачи.

## Две части
- **Общее для всех проектов** — `~/claude-work/system/` (не в этом репо): роли и протокол цепочки (`llm/`),
  аккаунты (`accounts/`), правила по инфре (`infra-rules/infra.md`), скрипты (`infra/`), дека (`agentdeck/`).
  Начать — `~/claude-work/system/README.md`.
- **Этот проект** — `.llm/` здесь: что за проект, его состояние и история, живые файлы цепочки.

## 📖 Порядок чтения
- **Интерактивная сессия с пользователем:** `AGENT_START.md` → `AGENT_START_ROUTINE.md` → по задаче.
- **Звено цепочки** (`<маш>-worker-N`, `<маш>-watcher-N`, `<маш>-fixer-N`, meta): своя рутина
  `~/claude-work/system/llm/routines/<РОЛЬ>_ROUTINE.md` (на yoga — ещё `…/llm/machines/YOGA_ROUTINE.md`),
  точка входа в проект — `chain/TASK.md`.

## 📁 Структура
```
v5chain/                         папка проекта: отсюда запускаются агенты и cron
├── .llm/
│   ├── README.md                этот файл
│   ├── AGENT_START.md           напутствие (интерактивные сессии)
│   ├── AGENT_START_ROUTINE.md   стартовая рутина (интерактивные сессии)
│   ├── AGENT_END_ROUTINE.md     финишная рутина (интерактивные сессии)
│   ├── .llmignore
│   ├── context/                 архитектура кода, сниппеты, зависимости (редко меняется)
│   ├── state/                   current / plan / issues / session_handoff / token_stats / YOGA_HANDOFF
│   ├── history/                 changelog_recent / changelog_archive / decisions (проект)
│   ├── tools/                   update_changelog.py
│   └── chain/                   «файлы цепочки»: пути в рутинах ролей вида TASK.md, knowledge/… — отсюда
│       ├── TASK.md              цель, правила, где что лежит (точка входа звена)
│       ├── STATUS.md PLAN.md ISSUES.md JOURNAL.md
│       ├── WATCHER_NOTES.md META_NOTES.md tick.log
│       ├── knowledge/           INDEX.md + факты проекта (один факт — одно место)
│       └── history/             decisions, token_stats, links_*, архивы, handoffs/
├── journals/                    пульс-журналы агентов и служебные файлы скриптов (_sentinel/), в .gitignore
├── .claude/usage_log.jsonl      телеметрия statusLine (пишется сама, в .gitignore)
└── docs/ src/ main.py …         код и документы проекта
```

## 🎯 Назначение файлов
| Файл | Частота изменений | Назначение |
|---|---|---|
| `context/*` | Редко | Базовые знания о коде |
| `state/*` | Каждую сессию | Актуальное состояние интерактивной работы |
| `history/*` | Append-only | История проекта |
| `chain/*` | По ходу цепочки | Состояние цепочки; правила ведения — `chain/TASK.md` §Как устроены знания |
| `AGENT_START.md` | По мере накопления опыта | Инсайты и антипаттерны |

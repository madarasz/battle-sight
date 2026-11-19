---
name: write-user-docs
description: Rules for the Battle Sight user documents README.md and docs/setup.md (non-technical reader, ASD-STE100, what goes in and what stays out, Work in Progress list). Use before you write or change README.md or docs/setup.md, also when a new feature, mode, key, overlay or setup step needs a user-facing line.
---

# Write the user documents

`README.md` and `docs/setup.md` are for non-technical users who install and run the app. Developer information goes in the `CLAUDE.md` files, `docs/dev/`, `docs/cli.md`, or `docs/testing.md`.

## README.md

It teaches how to use the app and lists its functions: how to start it, the workflow, what the overlay shows, and fixes for common problems. Write in ASD-STE100 Simplified Technical English. Keep it short and simple.

Do not put these in it:

- Game rules. The reader knows the game. Say what a function shows or does, not why the rule exists or how it works.
- Technical details: code internals, tests, config internals, module names, algorithms, tolerances, frame counts, development steps.
- Unnecessary details: where a minor element sits, exact colours of every mark, edge cases. One line per function is the aim.
- Obvious things. The user sees the UI: do not describe what is on it (panels, layouts, where a control is, what a list or the inspector shows, which buttons a page has). Tell only what the user cannot see: the order of steps, a hidden effect, the meaning of a colour or mark or icon.

Teach with UI elements: name the button, check box, list or menu item to use, never a key. No keys table: the README says once that the `h` key shows all key bindings.

A new function gets a short entry in the workflow ("Use the app"), and a line in "What you see" only when it draws a mark with a meaning that is not clear. Do not describe it in several places.

### Work in Progress list

`README.md` starts with a "Work in Progress" list: one line per priority with a status emoji (✅ done, ⏳ in progress or next, 🚧 not started). Update it when a priority starts or passes its done test. Keep each line short. Also update the priority table in the root `CLAUDE.md`.

### Setup

`README.md` "Setup" is only a short list that links to `docs/setup.md`.

## docs/setup.md

All setup and calibration steps go here, for the same non-technical reader and in the same language: install, print tags, place cards, camera, `data/board.toml` values a user sets, lens calibration, camera troubleshooting. When a change adds a setup step, add it here.

## Check before you finish

- Each new line names a UI element, not a key.
- No module name, config internal, tolerance or algorithm.
- The function is described in one place only.
- Sentences are short, one instruction per sentence, active voice.

# Operative copies

Specification for operatives that a team can field more than once. Status: steps 1 to 6 of section 9 done. Step 7 (sequence done test) waits for O3.

## 1. Problem

Some teams can field the same operative more than once (for example a Kommandos team with three Boys). The copies have the same name and the same miniature. Players paint them the same. We cannot ask players to paint or mark them differently.

The app must:

- keep a separate state for each copy (now: current wounds; later: dead, activated, orders),
- find the same operative type more than once in one snapshot,
- not mix up the copies,
- need as few user clicks as possible,
- let the user add and remove a copy,
- show the copies in `team.toml`.

## 2. Decisions

| ID | Decision |
|---|---|
| D1 | `team.toml` gives the maximum number of copies with `max = N` in the operative table (T1). |
| D2 | Appearance finds the operative **type** only. The gallery has one entry and one photo folder for each type. Copies never get their own photos. |
| D3 | The app tells copies apart by their **last known board position** (option B). The cost is the distance in board mm. |
| D4 | The app asks the user only when a swap is possible **and** the copies have different states. Copies with the same state can swap: the result is the same. |
| D5 | A copy keeps its number while it is in play. Removing Boy 2 does not change Boy 3. |
| D6 | Copy pairing is game state. It lives in `LiveSession`, not in a pipeline stage. The `image-pipeline` command and the `ident` stage give type labels only. |
| D7 | Tracking bases on the live video between snapshots (option C) is out of scope. Section 10 keeps it for later. |

## 3. team.toml

```toml
[operatives."Boy"]
base_mm = 32
wounds = 10
max = 4
```

- `max`: a whole number of 1 or more. Default 1. The loader rejects other values with a `ValueError` that names the file and the operative, as `check_wounds` does.
- An operative name must not contain `#`. The loader rejects it (`#` separates the copy number in a label).
- `operative-count` stays the team total. The app shows a notice when the in-play total is out of its range. It does not block the change.
- The file header comment gets one line for `max`.
- `ident.Operative` gets a field `max: int = 1`.

## 4. Labels

| Kind | Form | Use |
|---|---|---|
| Type label | `<team>/<operative>` | Gallery, photo folder, embeddings, `ident` stage, `<name>.ident.json`, benchmarks |
| Copy label | `<team>/<operative>#<k>`, k = 1, 2, ... | Live state: wounds, selection, target, team list, `in_play` |

- An operative with `max = 1` has no copy label. Its copy label is its type label. Wounds, `in_play`, ground truth files and baselines of existing teams do not change.
- New helpers in `ident.py`: `copy_label(type_label, k)`, `type_of(label)` (copy label to type label; a type label returns itself), `copy_number(label)` (None for a type label).
- Display name: `Boy 2` for a copy, `Boy` for an operative with `max = 1`. Overlay names, the team list, the name popup and notices use the display name.

## 5. Identification with copies

### 5.1 Type assignment

`ident.assign` gets one column for each copy in play, not one column for each type. The columns of the copies of one type have the same scores. The Hungarian step then can give a type to as many bases as there are copies of it in play, and not more.

The margin must compare only with **other types**. Today `margin` = assigned score minus the best other allowed column. With repeated columns the best other column is the same type, so the margin is 0 and every copy is unsure. Fix: take the best score of an allowed column of another type.

`Gallery.labels()` stays per type. A new argument gives the copy columns (for example `copies: dict[str, int]`, type label to number of copies in play). Without it `assign` works as now.

### 5.2 Copy pairing

After type assignment, `LiveSession` pairs the bases of each type to its copies. For each type T with more than one copy in play:

1. `B` = bases with type T in this snapshot. `C` = copies of T in play.
2. `LiveSession.positions: dict[str, tuple[float, float]]` holds the last known centre (board mm) of each copy.
3. Cost of base b and copy c: the distance from b to `positions[c]`. A copy without a position has the cost `copies.new_cost_mm` (config, default 150 mm).
4. Hungarian on the |B| x |C| cost matrix. Each base gets one copy. |B| <= |C| because of 5.1.
5. Unsure: for each two copies c1, c2 on bases b1, b2, the swapped total minus the best total is the swap cost. When the swap cost is below `copies.swap_margin_mm` (config, default 25 mm, about 1 in) **and** the two copies have a different state, both bases are unsure (yellow, `?`), as an unsure name today.
6. A copy with no base in this snapshot keeps its position and its state. It is not removed.

"Different state" = different current wounds. Later state fields (dead, activated, orders) join this test.

### 5.3 First snapshot

No copy has a position. All copies have the same state, so the order does not matter (D4). The app gives the numbers by board position: the base nearest to the top-left board corner gets the lowest free number, then the next. The order must be stable for the same board.

### 5.4 Position update

`positions[c]` changes:

- after each snapshot, for each copy with a base that is not unsure,
- when the user sets or swaps a name (6.1),
- when "Locate operative" (`base.add`) adds a base with a copy label.

Positions are not saved to a file (wounds are not saved either). A team change (`team.set`) clears the positions of that side.

## 6. UI

### 6.1 Name popup and swap

- The popup lists each copy of a type with its wounds, for example `Boy 1 (10/10)`, `Boy 2 (4/10)`.
- `base.choose(label)` with a copy label:
  - when another base holds that copy and that base has the same type: the two bases swap copies. Both bases become confirmed. Both positions change.
  - else: as today (the other base becomes unknown).
- The gallery photo of a fix goes to the type folder (`type_of(label)`). `LabelSession.choose` must pass the type label to `Gallery.add`.
- A swap is one click: right click the wrong base, pick the right copy.

### 6.2 Team list

- One row for each copy in play: `Boy 1`, `Boy 2`.
- `roster.add(slot, label)` with a type label adds the lowest free copy number of that type, up to `max`. With a copy label it adds that copy when it is free. Above `max`: an error notice.
- The "add operative" list of a side shows a type while its copies in play are less than `max`.
- `roster.remove(slot)` removes the selected copy. It drops the wounds and the position of that copy. When a removed copy is added again it starts with full wounds.
- When no copy of a type is in play, the type goes to `Gallery.out_of_play` and its base size leaves the known sizes, as today. While one copy is in play, both stay.
- `in_play` holds copy labels.

### 6.3 Start of a team

When a team is set, each type starts with **one** copy in play. The user adds more copies with the team list. Open point O1 keeps an automatic add.

### 6.4 State

- `state["bases"]` entries: `label` = copy label, new `type_label`, `copy` (number or None), `name` = display name.
- `state["rosters"]` entries: copy labels and display names, plus `max` and `copies_in_play` for each type.
- `docs/cli.md` gets the new fields. The JSON report of `image-pipeline` does not change (D6).

## 7. Config

New table `[copies]` in `data/board.toml`:

| Key | Default | Meaning |
|---|---|---|
| `swap_margin_mm` | 25 | Swap cost below this value with a different state gives unsure (5.2 step 5). |
| `new_cost_mm` | 150 | Pairing cost of a copy without a position (5.2 step 3). |

## 8. Benchmarks and tests

- `ident_bench.score` compares `type_of(found)` with the truth type label. Copy numbers do not count. The baseline does not move.
- `tools/annotate_bases.py` keeps writing type labels.
- Unit tests (no camera, no ML, `ColorEmbedder` or score matrices):
  - `team.toml` with `max`: load, bad values, `#` in a name.
  - `assign` with copy columns: a type goes to at most its number of copies; the margin compares other types only.
  - Pairing: two copies, one moves: no question. Two copies swap places with a different state: both unsure. Same swap with the same state: no question. A copy not found keeps position and wounds.
  - First snapshot order (5.3).
  - `base.choose` swap, `roster.add` up to `max`, `roster.remove` of one copy, wounds of each copy.
  - Layout tests in `tests/test_ui.py` for the copy rows and the popup.
- Done test (measurable): a sequence of at least 10 snapshots of a team with at least 2 copies of one type, with recorded copy truth. Of the bases with the right type, at least 95 % get the right copy without a click. Every wrong copy with a different state is shown as unsure. This needs a new truth format (a snapshot sequence) and a team with copies in the gallery: open point O3.

## 9. Work order

1. `team.toml` `max`, `Operative.max`, label helpers, loader checks, header comment, `docs/cli.md`.
2. `assign` copy columns and the margin fix. `ident_bench` with `type_of`.
3. `LiveSession`: `in_play` with copy labels, wounds per copy, display names, `roster.add` / `roster.remove` per copy, state fields.
4. Copy pairing, positions, unsure on a possible swap, first snapshot order.
5. Name popup with copies and the swap in `base.choose`. `LabelSession.choose` with the type folder.
6. Team list rows and the add list in the layouts. README "How to use" line for copies.
7. Sequence done test (after O3).

## 10. Later: option C (tracking on the live video)

When the done test shows too many wrong copies, add tracking between snapshots: follow each base on the live video (bases mapped with the live `BoardView`), see a base that goes away (lifted) and one that comes back, and move the copy with it. Risks: hands cover bases, a lifted model leaves the view, cost for each frame. It would feed `positions` only; sections 5 to 6 stay.

## 11. Open points

| ID | Point |
|---|---|
| O1 | Automatic add: when a snapshot finds more sure bases of a type than its copies in play and `max` allows more, add a copy with a notice. Saves clicks at the start. Risk: a false detection adds a copy. |
| O2 | First snapshot order: top-left distance or distance to the side's drop zone (needs a layout). |
| O3 | Test data: a team with copies (for example Kommandos) in the gallery, and a recorded snapshot sequence with copy truth. |
| O4 | One combined Hungarian (score and distance together) in place of two steps. Only if two steps give errors on O3 data. |
| O5 | A copy that is dead or out of action must not take a base. Needs the dead status first. |

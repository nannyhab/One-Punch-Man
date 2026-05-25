# ONE PUNCH MAN — Godot 4.6

Simple 2D arcade boxing. **Requires Godot 4.6.**

## Game modes

On launch, pick:

- **VS IRON MIKE AI** — one player vs computer
- **VS Player** — two players, same keyboard

## Controls

### Player 1 (blue, left)

| Key | Action |
|-----|--------|
| A / D | Move left / right |
| W / S | Ring depth |
| J | Straight — arm shoots forward |
| K | Hook — wide sweep from the side |
| L | Uppercut — crouch then glove rises up |
| Space | Guard |
| Left Shift | Dodge |

### Player 2 (red, right) — VS Player mode only

| Key | Action |
|-----|--------|
| Arrow keys | Move |
| U | Straight |
| I | Hook |
| O | Uppercut |
| M | Guard |
| Right Shift | Dodge |

## Open & play

1. Install [Godot 4.6](https://godotengine.org/download)
2. Open `project.godot`
3. Press **F5** → choose mode → fight

## Folder layout

```
scenes/   ModeSelect, Main, BoxingRing, PlayerBoxer, Player2Boxer, CPUBoxer, UI
scripts/  boxer, combat, player input, IRON MIKE AI, game state
```

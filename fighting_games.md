# Free macOS Fighting Games — Top 4 for *One Punch Man*

Native macOS, free, keyboard-controllable, three distinct punch inputs. Punches arrive on the host as classified events and are injected via `pynput`/`pyautogui` (or a direct socket for the custom build).

## Quick Comparison

| # | Game | macOS | License | 3 distinct punches? | Integration |
|---|---|---|---|---|---|
| 1 | **IKEMEN GO** | Native (Intel + Apple Silicon) | MIT | Yes — LP/MP/HP buttons | Easy: standard key events, windowed |
| 2 | **Tough Love Arena** | Browser + free Steam build | Freeware | Partial — Light/Heavy/Special | Trivial: focus tab, inject keys |
| 3 | **Custom Pygame/Godot fighter** | Native | MIT/MPL/yours | **Yes — you label them** | Trivial: socket bypasses keyboard layer |
| 4 | **HTML5 Punch-Out clones** | Browser | MIT / as-is | Forkable to yes | Trivial: edit JS key map |



## Per-Game Notes

### 1. IKEMEN GO — best primary target
Repo: `github.com/ikemen-engine/Ikemen-GO`. Open-source MUGEN-compatible engine in Go. Native Apple Silicon + Intel builds on Releases page. Standard 6-button scheme has **LP / MP / HP** as three discrete punch keys — cleanest match for your three classes. You can ship a custom character whose LP = straight, MP = hook, HP = uppercut so each gesture fires exactly the move it represents. Fully remappable, no anti-cheat, runs windowed. **Caveat:** stick to bundled Kung Fu Man / CC-licensed characters; most online MUGEN content is ripped from Capcom/SNK and not citeable in an academic deliverable.

### 2. Tough Love Arena — best zero-install demo
URL: `toughlovearena.com`. Browser-based, also a free Steam Mac build. Three attack buttons (Light/Heavy/Special) — one-to-one with your three classes, though not labeled "straight/hook/uppercut" visually. Remappable. Rollback netcode keeps frame timing rock-solid for live latency demos. **Caveat:** closed-source, so you can't add custom animations; browser tab must stay focused.

### 3. Custom Pygame / Godot sandbox — strongly recommended fallback
Templates: `SJUACM/Pygame-Demo` (minimal 2-player), `gamesketches/PyFighter` (data-driven move format), `panthavma/castagne` (Godot 4, MPL 2.0). **This is the hedge your slides already mention.** A 200-line Pygame script with animations literally named `straight.png`, `hook.png`, `uppercut.png` is a *stronger* academic demo because (a) gestures map 1:1 visibly, (b) you can pipe Xiao classifier output straight into the game over a Unix socket — bypassing `pyautogui` latency and macOS Accessibility-permission failures entirely, (c) zero licensing footnotes.

### 4. HTML5 Punch-Out clones — best thematic fit if you fork
Repos: `justin-austria/PunchOut`, `S-Stefan/punch-out-game`, `ben-denham/html5-boxing` (cleanest code). Punch-Out is the boxing-only reference with distinct strike types. Forking and renaming inputs to `J = Straight / K = Hook / L = Uppercut` is ~30 lines. **Caveat:** stale hobby repos (last commits 2–5 yrs old); ripped Nintendo sprites are copyrighted — replace assets or use the abstract `html5-boxing` art.

## Caveats

- **macOS Accessibility permission is mandatory** for `pyautogui`/`pynput`/AppleScript and is *per-binary* (Terminal, iTerm, VS Code each granted separately). Most common demo-day failure mode.
- **Keypress latency**: `pyautogui` defaults to ~110 ms on macOS (100 ms `PAUSE` + 10 ms `DARWIN_CATCH_UP_TIME`). Zero them out, or use `pynput` directly. For hard real-time, use the in-process socket path into Pygame/Godot.
- **Decision threshold**: if end-to-end latency >30 ms via keyboard injection, drop to the direct-socket Pygame path. If Accessibility permission can't be granted (managed/MDM laptop), use the browser route with JavaScript `KeyboardEvent` injection via Chrome DevTools Protocol.
- **The single highest-leverage upgrade**: IKEMEN GO + a 50-line custom `.cmd` file is one evening of work and gets you a real fighting-game look on the projector.

# ONE PUNCH MAN — Setup Guide

Follow these steps in order to get the game running on your machine. The whole process takes about 5–10 minutes if you do not already have Godot installed.

---

## Step 1: Get the project on your computer

Start by making sure you have the `boxing_2d` folder locally. This folder is the entire game — you do not need to install Python, Node, or anything else.

**If you are cloning from Git:**

```bash
git clone <repository-url>
cd <repository-name>/boxing_2d
```

**If someone sent you the files directly:**

Copy the `boxing_2d` folder anywhere on your machine (Desktop, Documents, etc.).

Before moving on, open the folder and confirm you see `project.godot` at the top level, along with `scenes/` and `scripts/` folders. If `project.godot` is missing, you are probably in the wrong directory — go one level deeper until you find it.

> **Important:** Open the `boxing_2d` folder in Godot, not the parent `Game dev` folder.

---

## Step 2: Download and install Godot 4.6

This game runs inside the Godot game engine. You only need the **Standard** version (not .NET).

1. Go to [https://godotengine.org/download](https://godotengine.org/download)
2. Download **Godot Engine 4.6**
3. Install it the way you normally install apps on your system:
   - **macOS:** Open the downloaded file and move `Godot.app` into Applications (or just run it from Downloads)
   - **Windows:** Run the installer, or unzip the download and double-click `Godot_v4.6-*.exe`
   - **Linux:** Extract the archive and run the `Godot` binary

Once Godot is installed, launch it. You should see the Project Manager — a screen that lists projects or lets you import a new one.

---

## Step 3: Import the project into Godot

With Godot open:

1. Click **Import** on the Project Manager screen
2. Navigate to the `boxing_2d` folder you set up in Step 1
3. Select the file **`project.godot`**
4. Click **Import & Edit**

Godot will open the project editor. The first time you import, Godot may spend a few seconds loading assets — wait until the editor is fully ready before continuing.

If you have opened this project before, you can skip the import step and just double-click **ONE PUNCH MAN** from the Project Manager list.

---

## Step 4: Run the game

Now that the project is open in the editor:

1. Press **F5** on your keyboard (or click the **Play** ▶ button in the top-right corner of the editor)
2. If Godot asks you to pick a main scene, choose **`res://scenes/ModeSelect.tscn`** and confirm
3. A game window should open showing the mode select screen

You are now running the game. To stop it, press **F8** or close the game window.

If you make changes to the code or scenes later, stop the game completely and press **F5** again to test fresh — do not rely on the in-game "Restart Round" button for that.

---

## Step 5: Choose a game mode

When the game starts, you land on the title / mode select screen.

1. Optionally type names for Player 1 and Player 2 in the text fields
2. Pick how you want to play:
   - Click **VS IRON MIKE AI** to fight the computer (one player)
   - Click **VS Player** for two people on the same keyboard

The fight scene loads automatically. You should see two boxers in a ring — blue on the left, red on the right — with health bars, stamina bars, and a 60-second timer at the top.

---

## Step 6: Play and test the game

Click inside the game window so it has keyboard focus, then try the controls below.

### If you chose VS IRON MIKE AI (Player 1 only)

Move with **A / D** (left/right) and **W / S** (up/down on the ring). Throw punches with **J** (straight), **K** (hook), and **L** (uppercut). Hold **Space** to guard and tap **Left Shift** to dodge.

Walk toward the CPU, throw a few punches, and watch for these signs that everything is working:

- The blue fighter moves when you press movement keys
- Each punch type looks different and briefly shows a label like `STRAIGHT` above your fighter
- Health and stamina bars change when you land or take hits
- The timer counts down from 60
- The match ends with a knockout or when time runs out

### If you chose VS Player (two players)

Player 1 (blue, left) uses the same keys as above. Player 2 (red, right) uses the **arrow keys** to move, **U / I / O** for straight / hook / uppercut, **M** to guard, and **Right Shift** to dodge.

Both fighters should respond independently. Try having each player move and punch to confirm both control schemes work.

### When the round ends

A result panel appears showing who won. From there you can click **Restart Round** to play again or **Main Menu** to go back to mode select.

---

## Step 7: Run from the terminal (optional)

If you prefer launching from a terminal, or want to script a quick smoke test, you can skip the editor and run Godot directly.

First, `cd` into the `boxing_2d` folder. Then run Godot with your platform's path:

**macOS:**

```bash
/Applications/Godot.app/Contents/MacOS/Godot --path . res://scenes/ModeSelect.tscn
```

**Windows:**

```cmd
"C:\Path\To\Godot_v4.6.exe" --path . res://scenes/ModeSelect.tscn
```

**Linux:**

```bash
./Godot --path . res://scenes/ModeSelect.tscn
```

Replace the Godot path with wherever you installed the engine. The game window should open the same way as pressing F5 in the editor.

---

## Step 8: Share a standalone build (optional)

If you want someone to play without installing Godot:

1. Open the project in Godot
2. Go to **Project → Export**
3. Add a preset for your target platform (Windows, macOS, or Linux)
4. Download export templates if Godot prompts you
5. Click **Export Project** and save the output file

Send them the exported executable — they can double-click it to play without any setup.

---

## If something goes wrong

**Godot will not import the project**
Make sure you selected `project.godot` inside the `boxing_2d` folder, not a parent directory. The folder must contain `scenes/` and `scripts/`.

**You get a version or "invalid project" error**
Install Godot **4.6** from the official site. This is a Godot 4 project and will not work with Godot 3.

**The game opens but fighters are missing or the screen is black**
Press **F8** to stop, then **F5** to run again. If that does not help, go to **Project → Reload Current Project** in the editor and check the **Output** panel at the bottom for red error messages.

**Keys do nothing**
Click inside the game window first. In VS Player mode, Player 2 keys only work for the red fighter on the right. On macOS, make sure the game window — not the editor behind it — is the active app.

**The window looks too small**
The game is built for 1280×720. Maximize the game window using your OS controls, or adjust the size under **Project → Project Settings → Display → Window** in the editor.

---

## Controls reference

**Player 1 (blue):** A/D move · W/S depth · J straight · K hook · L uppercut · Space guard · Left Shift dodge

**Player 2 (red, VS Player only):** Arrows move · U straight · I hook · O uppercut · M guard · Right Shift dodge

For a quick overview of the game itself, see [README.md](README.md).

Here's a summary of what was changed:

Files edited: All 4 character .cmd files

chars/kfm/kfm.cmd
chars/kfm720/kfm720.cmd
chars/kfm_zaxis/kfm.cmd
chars/kfm_zss/kfm.cmd
What was added to each file:
A new [Command] block was inserted that maps button c to the existing upper_x command name:


[Command]
name = "upper_x"
command = c
time = 1
This sits above the original motion entry which was kept untouched:


[Command]
name = "upper_x"
command = ~F, D, DF, x
Why this works: In Ikemen GO, multiple [Command] blocks with the same name all trigger the same move. So upper_x now fires from either the single button c or the full motion input.

The one in-game step you do once: In Options → Key Config, assign physical keyboard key F to button slot c — so pressing F in a fight sends button c to the engine, which triggers upper_x.

No engine source code was modified.

MAPPING AS OF NOW:
Left
Straight - A
Hook - S
Uppercut - F

Right
Straight - M
Hook - N
Uppercut - B
This game is downloaded from the IKEMEN GO github.
Download the right version for your OS here: https://github.com/ikemen-engine/Ikemen-GO/releases 

This repo automatically has the MacOs build. 
Once you download you can follow these steps



# MacOS

- if you have a Mac run these commands:


Make sure you are in the root directory of this project:
Then run these commands to give permissions 

```
xattr -d com.apple.quarantine Ikemen_GO-dev-macos/Ikemen_GO.command

```

```
chmod +x Ikemen_GO-dev-macos/Ikemen_GO.command

```

This command runs the app. Use this anytime you want to launch the game.
```
Ikemen_GO-dev-macos/Ikemen_GO.command
```


To play the game:
Go to either:
- 1. Practice
_You can just throw punches at an opponent who is stationary_

**- 2. VS MODE (preferred)**
**- select Quick Match**
**- play against Opponent 1 (the third guy)**
**_You play against computer_**


## Player selection and Key mappings
Player 1 (One Punch Man)
A - straight punch
S - hook
D - uppercut

Player 2 
M - straight punch
N - hook
B - uppercut

**Our main player is Player 1 (One Punch Man). You can use other players but Player 1 and Player 2 are the only ones with the necessary key mappings.**

Other moves are:
go forward -- right arrow
go backward -- left arrow
jump -- up arrow
crouch -- bottom arrow.
The game supports these moves but as of now for our ML project we are only mapping the Straight, Hook and Uppercut moves only and not movement moves.


extends Node

enum Mode { VS_CPU, VS_PLAYER }
enum MatchState { MENU, COUNTDOWN, FIGHTING, ROUND_OVER, MATCH_OVER }

var mode: Mode = Mode.VS_CPU
var match_state: MatchState = MatchState.MENU
var player1_name: String = "Player 1"
var player2_name: String = "Player 2"


func right_fighter_name() -> String:
	return "IRON MIKE AI" if mode == Mode.VS_CPU else player2_name


func is_match_active() -> bool:
	return match_state == MatchState.FIGHTING

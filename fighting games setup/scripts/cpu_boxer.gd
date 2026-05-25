class_name CPUBoxer
extends Boxer

var _think_timer: float = 0.0
var _move_dir: Vector2 = Vector2.ZERO

const THINK_EVERY := 0.3
const APPROACH_RANGE := 320.0
const HOLD_MIN_RANGE := 230.0
const HOLD_MAX_RANGE := 300.0
const RETREAT_RANGE := 205.0
const NO_PUNCH_RANGE := 190.0


func _physics_process(delta: float) -> void:
	if not GameState.is_match_active():
		_move_dir = Vector2.ZERO
		super._physics_process(delta)
		return
	if readability_locked:
		_move_dir = Vector2.ZERO
		super._physics_process(delta)
		return
	_think_timer -= delta
	if _think_timer <= 0.0:
		_think_timer = THINK_EVERY
		_decide()
	super._physics_process(delta)


func _decide() -> void:
	if not GameState.is_match_active():
		_move_dir = Vector2.ZERO
		return
	if opponent == null or combat_sm.is_stunned_or_down():
		_move_dir = Vector2.ZERO
		return

	var to_player := opponent.global_position - global_position
	var dist := absf(to_player.x)

	if stamina.is_low():
		_move_dir = Vector2(-signf(to_player.x), 0.0)
		combat_sm.stop_guard()
		return

	if opponent.combat_sm.state in [
		CombatStateMachine.BoxerState.PUNCH_STARTUP,
		CombatStateMachine.BoxerState.PUNCH_ACTIVE,
	]:
		if randf() < 0.6:
			combat_sm.start_guard()
			_move_dir = Vector2.ZERO
			return

	combat_sm.stop_guard()

	if dist < RETREAT_RANGE:
		_move_dir = Vector2(-signf(to_player.x), 0.0)
	elif dist > APPROACH_RANGE:
		_move_dir = Vector2(signf(to_player.x), 0.0)
	elif dist >= HOLD_MIN_RANGE and dist <= HOLD_MAX_RANGE and randf() < 0.45:
		_move_dir = Vector2.ZERO
		if dist < NO_PUNCH_RANGE:
			return
		var roll := randf()
		if roll < 0.5:
			try_punch(PunchData.Type.STRAIGHT)
		elif roll < 0.8:
			try_punch(PunchData.Type.HOOK)
		else:
			try_punch(PunchData.Type.UPPERCUT)
	else:
		_move_dir = Vector2.ZERO


func _get_move_input() -> Vector2:
	return _move_dir

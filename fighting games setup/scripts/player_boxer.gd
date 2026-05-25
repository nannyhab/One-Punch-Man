extends Boxer

@export var player_slot: int = 1


func _physics_process(delta: float) -> void:
	if GameState.is_match_active() and not readability_locked:
		_handle_input()
	super._physics_process(delta)


func _handle_input() -> void:
	if not GameState.is_match_active():
		return
	var pre := "p%d_" % player_slot
	if Input.is_action_just_pressed(pre + "punch_straight"):
		try_punch(PunchData.Type.STRAIGHT)
	elif Input.is_action_just_pressed(pre + "punch_hook"):
		try_punch(PunchData.Type.HOOK)
	elif Input.is_action_just_pressed(pre + "punch_uppercut"):
		try_punch(PunchData.Type.UPPERCUT)
	try_guard(Input.is_action_pressed(pre + "guard"))
	if Input.is_action_just_pressed(pre + "dodge"):
		var left := Input.is_action_pressed(pre + "move_left")
		try_dodge(-1 if left else 1)


func _get_move_input() -> Vector2:
	var pre := "p%d_" % player_slot
	var dir := Vector2.ZERO
	if Input.is_action_pressed(pre + "move_left"):
		dir.x -= 1.0
	if Input.is_action_pressed(pre + "move_right"):
		dir.x += 1.0
	if Input.is_action_pressed(pre + "move_up"):
		dir.y -= 1.0
	if Input.is_action_pressed(pre + "move_down"):
		dir.y += 1.0
	return dir.normalized() if dir != Vector2.ZERO else Vector2.ZERO

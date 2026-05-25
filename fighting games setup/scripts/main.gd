extends Node2D

const ROUND_TIME := 60.0
const DEBUG_SPACING := false
const HIT_STOP_CLEAN := 0.08
const HIT_STOP_BLOCK := 0.04
const POST_HIT_RESET_DISTANCE := 245.0
const POST_HIT_SEPARATION_SPEED := 900.0
const POST_HIT_CONTROL_LOCK := 0.18
const IDEAL_FIGHTING_DISTANCE := 240.0
const MIN_BODY_DISTANCE := 180.0

enum ReadabilityPhase { NONE, HIT_STOP, SEPARATING, RESET_GUARD }

@onready var ring: BoxingRing = $BoxingRing
@onready var player: Boxer = $PlayerBoxer
@onready var cpu: Boxer = $CPUBoxer
@onready var player2: Boxer = $Player2Boxer
@onready var camera: CameraShake = $Camera2D
@onready var ui = $UI

var right_fighter: Boxer
var vs_cpu: bool = true
var round_time_left: float = ROUND_TIME
var left_score: int = 0
var right_score: int = 0
var round_active: bool = true
var readability_phase: int = ReadabilityPhase.NONE
var readability_timer: float = 0.0
var readability_duration: float = 0.0
var left_start_x: float = 0.0
var right_start_x: float = 0.0
var left_target_x: float = 0.0
var right_target_x: float = 0.0


func _ready() -> void:
	GameState.match_state = GameState.MatchState.FIGHTING
	vs_cpu = GameState.mode == GameState.Mode.VS_CPU
	if vs_cpu:
		cpu.visible = true
		cpu.set_physics_process(true)
		player2.visible = false
		player2.set_physics_process(false)
		right_fighter = cpu
	else:
		cpu.visible = false
		cpu.set_physics_process(false)
		player2.visible = true
		player2.set_physics_process(true)
		right_fighter = player2

	player.setup(right_fighter, ring)
	right_fighter.setup(player, ring)
	player.global_position = Vector2(400, 360)
	right_fighter.global_position = Vector2(880, 360)

	ui.setup_fight(player, right_fighter, vs_cpu, GameState.player1_name, GameState.right_fighter_name())
	ui.restart_requested.connect(_restart_round)
	ui.menu_requested.connect(_go_to_menu)

	player.combat_event.connect(_on_combat_event)
	right_fighter.combat_event.connect(_on_combat_event)
	player.punch_started.connect(_on_punch_started)
	right_fighter.punch_started.connect(_on_punch_started)
	player.health.knocked_out.connect(_on_knockout.bind(true))
	right_fighter.health.knocked_out.connect(_on_knockout.bind(false))

	ui.update_timer(round_time_left)
	ui.update_score(left_score, right_score)


func _process(delta: float) -> void:
	if GameState.is_match_active():
		_tick_readability_reset(delta)
	_update_debug_spacing()
	if not GameState.is_match_active():
		return
	round_time_left -= delta
	ui.update_timer(maxf(0.0, round_time_left))
	if round_time_left <= 0.0:
		_end_round_by_timer()


func _physics_process(_delta: float) -> void:
	if GameState.is_match_active() and right_fighter:
		_enforce_readable_distance()


func _on_combat_event(result: CombatStateMachine.HitResult, punch_name: String, attacker_is_left: bool) -> void:
	if not GameState.is_match_active():
		return
	ui.show_feedback(_format_feedback(result, punch_name))
	match result:
		CombatStateMachine.HitResult.CLEAN_HIT:
			if attacker_is_left:
				left_score += 10
			else:
				right_score += 10
			camera.shake(1.5)
		CombatStateMachine.HitResult.COUNTER_HIT:
			if attacker_is_left:
				left_score += 20
			else:
				right_score += 20
			camera.shake(2.5)
		CombatStateMachine.HitResult.BLOCKED:
			if attacker_is_left:
				left_score += 2
			else:
				right_score += 2
	if result != CombatStateMachine.HitResult.WHIFF \
			and player.health.current_health > 0 \
			and right_fighter.health.current_health > 0:
		_start_readability_reset(result, punch_name, attacker_is_left)
	ui.update_score(left_score, right_score)
	print("[SCORE]", left_score, right_score)


func _on_punch_started(punch_name: String, attacker_is_left: bool) -> void:
	if not GameState.is_match_active():
		return
	ui.show_move(punch_name, attacker_is_left)


func _format_feedback(result: CombatStateMachine.HitResult, punch_name: String) -> String:
	match result:
		CombatStateMachine.HitResult.CLEAN_HIT:
			return "Clean %s" % punch_name
		CombatStateMachine.HitResult.BLOCKED:
			return "Blocked %s" % punch_name
		CombatStateMachine.HitResult.COUNTER_HIT:
			return "Counter %s" % punch_name
		CombatStateMachine.HitResult.WHIFF:
			return "Whiff"
	return punch_name


func _start_readability_reset(
	result: CombatStateMachine.HitResult,
	punch_name: String,
	attacker_is_left: bool
) -> void:
	if not GameState.is_match_active():
		return
	if readability_phase != ReadabilityPhase.NONE:
		_finish_readability_reset()
	var hit_stop := HIT_STOP_BLOCK if result == CombatStateMachine.HitResult.BLOCKED else HIT_STOP_CLEAN
	_start_phase(ReadabilityPhase.HIT_STOP, hit_stop)
	player.set_readability_lock(true, true)
	right_fighter.set_readability_lock(true, true)
	_set_separation_targets(punch_name, attacker_is_left)


func _tick_readability_reset(delta: float) -> void:
	if readability_phase == ReadabilityPhase.NONE:
		return
	readability_timer += delta
	match readability_phase:
		ReadabilityPhase.HIT_STOP:
			if readability_timer >= readability_duration:
				player.set_readability_lock(true, false)
				right_fighter.set_readability_lock(true, false)
				left_start_x = player.global_position.x
				right_start_x = right_fighter.global_position.x
				_start_phase(ReadabilityPhase.SEPARATING, 999.0)
		ReadabilityPhase.SEPARATING:
			var step := POST_HIT_SEPARATION_SPEED * delta
			player.global_position.x = move_toward(player.global_position.x, left_target_x, step)
			right_fighter.global_position.x = move_toward(right_fighter.global_position.x, right_target_x, step)
			player.global_position = ring.clamp_position(player.global_position)
			right_fighter.global_position = ring.clamp_position(right_fighter.global_position)
			_enforce_readable_distance()
			var target_reached := is_equal_approx(player.global_position.x, left_target_x) \
					and is_equal_approx(right_fighter.global_position.x, right_target_x)
			var readable_dist := right_fighter.global_position.x - player.global_position.x
			if target_reached and readable_dist >= IDEAL_FIGHTING_DISTANCE:
				player.force_guard_reset()
				right_fighter.force_guard_reset()
				_start_phase(ReadabilityPhase.RESET_GUARD, POST_HIT_CONTROL_LOCK)
		ReadabilityPhase.RESET_GUARD:
			_enforce_readable_distance()
			if readability_timer >= readability_duration:
				_finish_readability_reset()


func _start_phase(phase: int, duration: float) -> void:
	readability_phase = phase
	readability_timer = 0.0
	readability_duration = maxf(0.01, duration)


func _finish_readability_reset() -> void:
	readability_phase = ReadabilityPhase.NONE
	readability_timer = 0.0
	if not GameState.is_match_active():
		return
	player.set_readability_lock(false, false)
	right_fighter.set_readability_lock(false, false)
	player.force_guard_reset()
	right_fighter.force_guard_reset()
	_enforce_readable_distance()


func _set_separation_targets(punch_name: String, attacker_is_left: bool) -> void:
	var distance := _target_distance_for_punch(punch_name)
	if attacker_is_left:
		left_target_x = player.global_position.x - 8.0
		right_target_x = left_target_x + distance
	else:
		right_target_x = right_fighter.global_position.x + 8.0
		left_target_x = right_target_x - distance
	_adjust_targets_inside_ring(distance)


func _target_distance_for_punch(punch_name: String) -> float:
	return POST_HIT_RESET_DISTANCE


func _adjust_targets_inside_ring(distance: float) -> void:
	var left_limit := ring.ring_rect.position.x + 40.0
	var right_limit := ring.ring_rect.end.x - 40.0
	if left_target_x < left_limit:
		left_target_x = left_limit
		right_target_x = left_target_x + distance
	if right_target_x > right_limit:
		right_target_x = right_limit
		left_target_x = right_target_x - distance


func _enforce_readable_distance() -> void:
	var dist := right_fighter.global_position.x - player.global_position.x
	if dist >= MIN_BODY_DISTANCE:
		return
	var midpoint := (player.global_position.x + right_fighter.global_position.x) * 0.5
	player.global_position.x = midpoint - MIN_BODY_DISTANCE * 0.5
	right_fighter.global_position.x = midpoint + MIN_BODY_DISTANCE * 0.5
	player.global_position = ring.clamp_position(player.global_position)
	right_fighter.global_position = ring.clamp_position(right_fighter.global_position)
	dist = right_fighter.global_position.x - player.global_position.x
	if dist < MIN_BODY_DISTANCE:
		if player.global_position.x <= ring.ring_rect.position.x + 45.0:
			right_fighter.global_position.x = player.global_position.x + MIN_BODY_DISTANCE
		else:
			player.global_position.x = right_fighter.global_position.x - MIN_BODY_DISTANCE
		player.global_position = ring.clamp_position(player.global_position)
		right_fighter.global_position = ring.clamp_position(right_fighter.global_position)


func _update_debug_spacing() -> void:
	var dist := absf(right_fighter.global_position.x - player.global_position.x) if right_fighter else 0.0
	ui.update_debug_distance(dist, DEBUG_SPACING)
	if DEBUG_SPACING:
		queue_redraw()


func _draw() -> void:
	if not DEBUG_SPACING or right_fighter == null:
		return
	draw_line(player.global_position, right_fighter.global_position, Color(1, 1, 0.4, 0.28), 2.0)


func _on_knockout(left_ko: bool) -> void:
	if not GameState.is_match_active():
		return
	if left_ko:
		right_score += 50
		_end_match("ONE PUNCH MAN\n" + _right_name() + " Wins — Knockout!")
	else:
		left_score += 50
		_end_match("ONE PUNCH MAN\n" + GameState.player1_name + " Wins — Knockout!")
	ui.update_score(left_score, right_score)


func _end_round_by_timer() -> void:
	if not GameState.is_match_active():
		return
	if left_score > right_score:
		_end_match("ONE PUNCH MAN\n" + GameState.player1_name + " Wins — Decision!")
	elif right_score > left_score:
		_end_match("ONE PUNCH MAN\n" + _right_name() + " Wins — Decision!")
	else:
		_end_match("ONE PUNCH MAN\nDraw!")


func _end_match(result_text: String) -> void:
	GameState.match_state = GameState.MatchState.MATCH_OVER
	round_active = false
	readability_phase = ReadabilityPhase.NONE
	readability_timer = 0.0
	_disable_all_hitboxes()
	player.freeze_for_match_end()
	right_fighter.freeze_for_match_end()
	cpu.set_physics_process(false)
	player2.set_physics_process(false)
	ui.show_result(result_text)


func _disable_all_hitboxes() -> void:
	player._end_all_hitboxes()
	right_fighter._end_all_hitboxes()


func _right_name() -> String:
	return GameState.right_fighter_name()


func _restart_round() -> void:
	get_tree().reload_current_scene()


func _go_to_menu() -> void:
	get_tree().change_scene_to_file("res://scenes/ModeSelect.tscn")

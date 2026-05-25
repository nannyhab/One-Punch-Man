class_name CombatStateMachine
extends RefCounted

enum BoxerState {
	IDLE,
	MOVING,
	GUARDING,
	DODGING,
	PUNCH_STARTUP,
	PUNCH_ACTIVE,
	PUNCH_RECOVERY,
	HIT_REACT,
	BLOCK_REACT,
	STUNNED,
	KNOCKDOWN,
}

enum HitResult {
	CLEAN_HIT,
	BLOCKED,
	WHIFF,
	COUNTER_HIT,
}

var state: BoxerState = BoxerState.IDLE
var current_punch: PunchData.Type = PunchData.Type.STRAIGHT
var dodge_cooldown: float = 0.0
var stun_timer: float = 0.0
var hit_react_timer: float = 0.0
var block_react_timer: float = 0.0
var punch_phase_timer: float = 0.0
var punch_has_hit: bool = false

const DODGE_COOLDOWN_TIME := 0.6
const STUN_DURATION := 1.2
const HIT_REACT_DURATION := 0.35
const BLOCK_REACT_DURATION := 0.25


func can_move() -> bool:
	match state:
		BoxerState.IDLE, BoxerState.MOVING, BoxerState.GUARDING:
			return true
		BoxerState.PUNCH_RECOVERY:
			return false
	return false


func can_punch() -> bool:
	match state:
		BoxerState.IDLE, BoxerState.MOVING, BoxerState.GUARDING:
			return true
	return false


func can_guard() -> bool:
	match state:
		BoxerState.IDLE, BoxerState.MOVING:
			return true
	return false


func can_dodge() -> bool:
	return dodge_cooldown <= 0.0 and state in [BoxerState.IDLE, BoxerState.MOVING, BoxerState.GUARDING]


func start_punch(punch_type: PunchData.Type) -> bool:
	if not can_punch():
		return false
	current_punch = punch_type
	punch_has_hit = false
	_set_state(BoxerState.PUNCH_STARTUP)
	punch_phase_timer = 0.0
	return true


func start_guard() -> void:
	if can_guard():
		_set_state(BoxerState.GUARDING)


func force_guard() -> void:
	punch_phase_timer = 0.0
	punch_has_hit = false
	hit_react_timer = 0.0
	block_react_timer = 0.0
	stun_timer = 0.0
	_set_state(BoxerState.GUARDING)


func stop_guard() -> void:
	if state == BoxerState.GUARDING:
		_set_state(BoxerState.IDLE)


func start_dodge(_direction: int) -> bool:
	if not can_dodge():
		return false
	dodge_cooldown = DODGE_COOLDOWN_TIME
	_set_state(BoxerState.DODGING)
	punch_phase_timer = 0.0
	return true


func apply_hit_react() -> void:
	_set_state(BoxerState.HIT_REACT)
	hit_react_timer = HIT_REACT_DURATION


func apply_block_react() -> void:
	_set_state(BoxerState.BLOCK_REACT)
	block_react_timer = BLOCK_REACT_DURATION


func apply_stun() -> void:
	_set_state(BoxerState.STUNNED)
	stun_timer = STUN_DURATION


func apply_knockdown() -> void:
	_set_state(BoxerState.KNOCKDOWN)


func tick(delta: float, punch_data: PunchData) -> void:
	if dodge_cooldown > 0.0:
		dodge_cooldown -= delta

	match state:
		BoxerState.PUNCH_STARTUP:
			punch_phase_timer += delta
			if punch_phase_timer >= punch_data.startup:
				_set_state(BoxerState.PUNCH_ACTIVE)
				punch_phase_timer = 0.0

		BoxerState.PUNCH_ACTIVE:
			punch_phase_timer += delta
			if punch_phase_timer >= punch_data.active:
				_set_state(BoxerState.PUNCH_RECOVERY)
				punch_phase_timer = 0.0

		BoxerState.PUNCH_RECOVERY:
			punch_phase_timer += delta
			if punch_phase_timer >= punch_data.recovery:
				_set_state(BoxerState.IDLE)
				punch_phase_timer = 0.0

		BoxerState.DODGING:
			punch_phase_timer += delta
			if punch_phase_timer >= 0.25:
				_set_state(BoxerState.IDLE)
				punch_phase_timer = 0.0

		BoxerState.HIT_REACT:
			hit_react_timer -= delta
			if hit_react_timer <= 0.0:
				_set_state(BoxerState.IDLE)

		BoxerState.BLOCK_REACT:
			block_react_timer -= delta
			if block_react_timer <= 0.0:
				_set_state(BoxerState.IDLE)

		BoxerState.STUNNED:
			stun_timer -= delta
			if stun_timer <= 0.0:
				_set_state(BoxerState.IDLE)


func is_punch_active() -> bool:
	return state == BoxerState.PUNCH_ACTIVE


func is_guarding() -> bool:
	return state == BoxerState.GUARDING


func is_dodging() -> bool:
	return state == BoxerState.DODGING


func is_stunned_or_down() -> bool:
	return state in [BoxerState.STUNNED, BoxerState.KNOCKDOWN]


func set_moving(is_moving: bool) -> void:
	if state in [BoxerState.IDLE, BoxerState.MOVING]:
		_set_state(BoxerState.MOVING if is_moving else BoxerState.IDLE)


func _set_state(new_state: BoxerState) -> void:
	state = new_state

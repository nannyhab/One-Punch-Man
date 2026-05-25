class_name Boxer
extends CharacterBody2D

signal combat_event(result: CombatStateMachine.HitResult, punch_name: String, attacker_is_player: bool)
signal punch_started(punch_name: String, attacker_is_player: bool)

@export var is_player: bool = true

@export var move_speed_x: float = 220.0
@export var move_speed_y: float = 90.0
@export var acceleration: float = 1400.0
@export var friction: float = 1800.0
@export var min_body_distance: float = 180.0
@export var ideal_fighting_distance: float = 240.0

var opponent: Boxer = null
var ring: BoxingRing = null
var facing: int = 1
var combat_sm := CombatStateMachine.new()
var punch_library: Dictionary = {}
var current_punch_data: PunchData = null
var last_punch_type: PunchData.Type = PunchData.Type.STRAIGHT
var readability_locked: bool = false

const VISUAL_SCALE := 1.42
const DEBUG_COMBAT_TEST := false

@onready var health: HealthComponent = $HealthComponent
@onready var stamina: StaminaComponent = $StaminaComponent
@onready var visual: Node2D = $Visual
@onready var anim: AnimationPlayer = $Visual/AnimationPlayer
@onready var straight_hitbox: Hitbox = $Hitboxes/StraightHitbox
@onready var hook_hitbox: Hitbox = $Hitboxes/HookHitbox
@onready var uppercut_hitbox: Hitbox = $Hitboxes/UppercutHitbox


func _ready() -> void:
	punch_library = PunchData.create_defaults()
	for hb in [straight_hitbox, hook_hitbox, uppercut_hitbox]:
		hb.setup(self, is_player)
		hb.hit_connected.connect(_on_hitbox_connected)
	for child in $Hurtboxes.get_children():
		if child is Hurtbox:
			child.setup(self, is_player)
	_play_anim("idle_guard")


func setup(opponent_ref: Boxer, ring_ref: BoxingRing) -> void:
	opponent = opponent_ref
	ring = ring_ref
	facing = 1 if is_player else -1
	visual.scale = Vector2(VISUAL_SCALE * facing, VISUAL_SCALE)
	_apply_facing()


func _apply_facing() -> void:
	visual.scale = Vector2(VISUAL_SCALE * facing, VISUAL_SCALE)


func _physics_process(delta: float) -> void:
	_update_facing()

	if not GameState.is_match_active():
		velocity = Vector2.ZERO
		_end_all_hitboxes()
		return

	if readability_locked:
		velocity = Vector2.ZERO
		_end_all_hitboxes()
		return

	if current_punch_data:
		var prev_state := combat_sm.state
		combat_sm.tick(delta, current_punch_data)
		if prev_state == CombatStateMachine.BoxerState.PUNCH_RECOVERY \
				and combat_sm.state == CombatStateMachine.BoxerState.IDLE \
				and not combat_sm.punch_has_hit:
			combat_event.emit(CombatStateMachine.HitResult.WHIFF, current_punch_data.display_name, is_player)
			current_punch_data = null
	else:
		combat_sm.tick(delta, punch_library[PunchData.Type.STRAIGHT])

	_update_hitboxes()
	_handle_movement(delta)
	move_and_slide()
	_clamp_to_ring()
	_prevent_overlap()
	_update_animation()


func _update_facing() -> void:
	if opponent == null:
		return
	facing = 1 if global_position.x < opponent.global_position.x else -1
	_apply_facing()


func _handle_movement(delta: float) -> void:
	if not combat_sm.can_move():
		velocity = velocity.move_toward(Vector2.ZERO, friction * delta)
		return

	var speed_scale := 0.72 if stamina.is_exhausted() else 1.0
	var input_dir := _get_move_input()
	input_dir = _block_forward_overlap(input_dir)
	if input_dir != Vector2.ZERO:
		var target := input_dir * Vector2(move_speed_x * speed_scale, move_speed_y * speed_scale)
		velocity = velocity.move_toward(target, acceleration * delta)
		combat_sm.set_moving(true)
	else:
		velocity = velocity.move_toward(Vector2.ZERO, friction * delta)
		combat_sm.set_moving(combat_sm.state == CombatStateMachine.BoxerState.GUARDING)


func _get_move_input() -> Vector2:
	return Vector2.ZERO


func _clamp_to_ring() -> void:
	if ring:
		global_position = ring.clamp_position(global_position)


func _prevent_overlap() -> void:
	if opponent == null:
		return
	var left_boxer := self if global_position.x <= opponent.global_position.x else opponent
	var right_boxer := opponent if left_boxer == self else self
	var dist_x := right_boxer.global_position.x - left_boxer.global_position.x
	if dist_x < min_body_distance:
		var midpoint := (left_boxer.global_position.x + right_boxer.global_position.x) * 0.5
		left_boxer.global_position.x = midpoint - min_body_distance * 0.5
		right_boxer.global_position.x = midpoint + min_body_distance * 0.5
		left_boxer._clamp_to_ring()
		right_boxer._clamp_to_ring()

		# If one fighter was clamped by the ropes, preserve spacing by moving the other fighter.
		var corrected_dist := right_boxer.global_position.x - left_boxer.global_position.x
		if ring and corrected_dist < min_body_distance:
			if left_boxer.global_position.x <= ring.ring_rect.position.x + 45.0:
				right_boxer.global_position.x = left_boxer.global_position.x + min_body_distance
			else:
				left_boxer.global_position.x = right_boxer.global_position.x - min_body_distance
		velocity.x = 0.0
		opponent.velocity.x = 0.0
		if combat_sm.can_guard():
			combat_sm.start_guard()
		if opponent.combat_sm.can_guard():
			opponent.combat_sm.start_guard()
		_clamp_to_ring()
		opponent._clamp_to_ring()


func set_readability_lock(locked: bool, freeze_animation: bool = false) -> void:
	readability_locked = locked
	velocity = Vector2.ZERO
	_end_all_hitboxes()
	if anim:
		anim.speed_scale = 0.0 if freeze_animation else 1.0


func force_guard_reset() -> void:
	current_punch_data = null
	combat_sm.force_guard()
	_play_anim("guard")
	if anim:
		anim.speed_scale = 1.0


func _end_all_hitboxes() -> void:
	for hb in [straight_hitbox, hook_hitbox, uppercut_hitbox]:
		hb.end_punch_window()


func freeze_for_match_end() -> void:
	readability_locked = true
	velocity = Vector2.ZERO
	current_punch_data = null
	_end_all_hitboxes()
	combat_sm.force_guard()
	_play_anim("guard")
	if anim:
		anim.speed_scale = 1.0


func _block_forward_overlap(input_dir: Vector2) -> Vector2:
	if opponent == null or input_dir.x == 0.0:
		return input_dir
	var dist_x := absf(global_position.x - opponent.global_position.x)
	var moving_toward_opponent := signf(input_dir.x) == float(facing)
	if moving_toward_opponent and dist_x <= min_body_distance + 12.0:
		input_dir.x = 0.0
	return input_dir


func try_punch(punch_type: PunchData.Type) -> bool:
	if not GameState.is_match_active():
		return false
	if not combat_sm.can_punch():
		return false
	var data: PunchData = punch_library[punch_type]
	if not stamina.spend(data.stamina_cost):
		return false
	if not combat_sm.start_punch(punch_type):
		stamina.current_stamina += data.stamina_cost
		return false
	current_punch_data = data
	last_punch_type = punch_type
	punch_started.emit(data.display_name, is_player)
	_aim_visual_punch(punch_type)
	_play_punch_animation(punch_type)
	return true


func try_guard(pressed: bool) -> void:
	if not GameState.is_match_active():
		return
	if pressed:
		combat_sm.start_guard()
		_play_anim("guard")
	else:
		combat_sm.stop_guard()


func try_dodge(direction: int) -> bool:
	if not GameState.is_match_active():
		return false
	if not combat_sm.can_dodge() or not stamina.spend(9.0):
		return false
	if not combat_sm.start_dodge(direction):
		stamina.current_stamina += 9.0
		return false
	_play_anim("dodge_left" if direction < 0 else "dodge_right")
	return true


func _update_hitboxes() -> void:
	if not GameState.is_match_active():
		_end_all_hitboxes()
		return
	var punch_active := combat_sm.is_punch_active() and current_punch_data != null
	if DEBUG_COMBAT_TEST and opponent:
		print("[COMBAT]", name, " state=", combat_sm.state, " punch=", combat_sm.current_punch, " dist=", absf(global_position.x - opponent.global_position.x))
	var active_hb := _get_active_hitbox() if punch_active else null
	for hb in [straight_hitbox, hook_hitbox, uppercut_hitbox]:
		if hb == active_hb:
			var hitbox_offset := _active_visual_hitbox_offset(current_punch_data)
			hb.update_position_and_shape(
				hitbox_offset,
				current_punch_data.hitbox_size,
				facing
			)
			if not hb.active:
				hb.begin_punch_window()
			else:
				hb.refresh_overlaps()
		else:
			hb.end_punch_window()


func _get_active_hitbox() -> Hitbox:
	match combat_sm.current_punch:
		PunchData.Type.STRAIGHT: return straight_hitbox
		PunchData.Type.HOOK: return hook_hitbox
		PunchData.Type.UPPERCUT: return uppercut_hitbox
	return null


func _active_visual_hitbox_offset(data: PunchData) -> Vector2:
	if visual and visual.has_method("get_punch_hitbox_offset"):
		return visual.get_punch_hitbox_offset(data.punch_type)
	return data.hitbox_offset


func _on_hitbox_connected(hurtbox: Hurtbox) -> void:
	if not GameState.is_match_active():
		return
	if not combat_sm.is_punch_active() or current_punch_data == null:
		return
	if hurtbox.owner_boxer == self or combat_sm.punch_has_hit:
		return
	combat_sm.punch_has_hit = true
	_resolve_hit(hurtbox)
	_end_all_hitboxes()


func _resolve_hit(hurtbox: Hurtbox) -> void:
	if not GameState.is_match_active():
		return
	var punch_data := current_punch_data
	if punch_data == null:
		return
	var target := hurtbox.owner_boxer
	var result := CombatStateMachine.HitResult.CLEAN_HIT
	var damage := punch_data.damage
	var punch_name := punch_data.display_name

	if target.combat_sm.is_dodging():
		result = CombatStateMachine.HitResult.WHIFF
	elif target.combat_sm.is_guarding():
		result = CombatStateMachine.HitResult.BLOCKED
		damage = 1
	elif target.combat_sm.state in [
		CombatStateMachine.BoxerState.PUNCH_STARTUP,
		CombatStateMachine.BoxerState.PUNCH_ACTIVE,
	]:
		result = CombatStateMachine.HitResult.COUNTER_HIT
		damage += punch_data.counter_bonus_damage

	if result == CombatStateMachine.HitResult.WHIFF:
		print("[HIT]", _display_name(), punch_name, "->", target._display_name(), _hit_result_name(result))
		combat_event.emit(result, punch_name, is_player)
		return

	if result == CombatStateMachine.HitResult.BLOCKED:
		target.combat_sm.apply_block_react()
		target._play_anim("block_react")
		target.health.take_damage(damage)
		target.stamina.drain(_stamina_damage_for_hit(damage, result), true)
	elif result == CombatStateMachine.HitResult.COUNTER_HIT:
		target.combat_sm.apply_stun()
		target._play_anim("stunned")
		target.health.take_damage(damage)
		target.stamina.drain(_stamina_damage_for_hit(damage, result), true)
	else:
		target.combat_sm.apply_hit_react()
		target._play_anim("hit_body" if hurtbox.hurt_type == Hurtbox.HurtType.BODY else "hit_head")
		target.health.take_damage(damage)
		target.stamina.drain(_stamina_damage_for_hit(damage, result), true)

	if target.health.current_health <= 0:
		target.combat_sm.apply_knockdown()
		target._play_anim("knockdown")
	print("[HIT]", _display_name(), punch_name, "->", target._display_name(), _hit_result_name(result))
	combat_event.emit(result, punch_name, is_player)


func _stamina_damage_for_hit(_damage: int, result: CombatStateMachine.HitResult) -> float:
	match result:
		CombatStateMachine.HitResult.BLOCKED:
			return 4.0
		CombatStateMachine.HitResult.COUNTER_HIT:
			return 8.0
	return 8.0


func _update_animation() -> void:
	if combat_sm.state in [
		CombatStateMachine.BoxerState.PUNCH_STARTUP,
		CombatStateMachine.BoxerState.PUNCH_ACTIVE,
		CombatStateMachine.BoxerState.PUNCH_RECOVERY,
		CombatStateMachine.BoxerState.HIT_REACT,
		CombatStateMachine.BoxerState.BLOCK_REACT,
		CombatStateMachine.BoxerState.STUNNED,
		CombatStateMachine.BoxerState.KNOCKDOWN,
		CombatStateMachine.BoxerState.DODGING,
	]:
		return
	if combat_sm.is_guarding():
		_play_anim("guard")
	elif velocity.length() > 10.0:
		var toward := signf(velocity.x) == float(facing)
		_play_anim("move_forward" if toward else "move_backward")
	else:
		_play_anim("idle_guard")


func _play_punch_animation(punch_type: PunchData.Type) -> void:
	match punch_type:
		PunchData.Type.STRAIGHT: _play_anim("straight")
		PunchData.Type.HOOK: _play_anim("hook")
		PunchData.Type.UPPERCUT: _play_anim("uppercut")


func _aim_visual_punch(punch_type: PunchData.Type) -> void:
	if opponent == null or not visual.has_method("aim_next_punch"):
		return
	var target_world := opponent.get_target_point(punch_type, facing)
	var target_local := _target_to_facing_local(target_world)
	visual.aim_next_punch(punch_type, target_local, facing)


func _target_to_facing_local(target_world: Vector2) -> Vector2:
	var relative := target_world - global_position
	var facing_x := maxf(relative.x * float(facing), 0.0)
	return Vector2(facing_x, relative.y) / VISUAL_SCALE


func get_target_point(punch_type: PunchData.Type, attacker_facing: int = 1) -> Vector2:
	match punch_type:
		PunchData.Type.HOOK:
			return global_position + Vector2(-attacker_facing * 18.0, -86.0)
		PunchData.Type.UPPERCUT:
			return global_position + Vector2(0.0, -96.0)
	return global_position + Vector2(0.0, -92.0)


func _display_name() -> String:
	return GameState.player1_name if is_player else GameState.right_fighter_name()


func _hit_result_name(result: CombatStateMachine.HitResult) -> String:
	match result:
		CombatStateMachine.HitResult.CLEAN_HIT:
			return "CLEAN_HIT"
		CombatStateMachine.HitResult.BLOCKED:
			return "BLOCKED"
		CombatStateMachine.HitResult.COUNTER_HIT:
			return "COUNTER_HIT"
		CombatStateMachine.HitResult.WHIFF:
			return "WHIFF"
	return "UNKNOWN"


func _play_anim(anim_name: String) -> void:
	if anim and anim.has_animation(anim_name):
		anim.play(anim_name)

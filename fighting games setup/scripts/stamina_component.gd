class_name StaminaComponent
extends Node

signal stamina_changed(current: float, maximum: float)

@export var max_stamina: float = 100.0
@export var regen_rate: float = 9.0
@export var low_threshold: float = 25.0
@export var min_to_act: float = 1.0

var current_stamina: float = 100.0
var _regen_delay: float = 0.0

const REGEN_DELAY_AFTER_USE := 0.55
const REGEN_DELAY_AFTER_HIT := 1.0


func _ready() -> void:
	current_stamina = max_stamina


func _process(delta: float) -> void:
	if not GameState.is_match_active():
		return
	if current_stamina >= max_stamina:
		return
	var rate := regen_rate
	if _regen_delay > 0.0:
		_regen_delay -= delta
		rate *= 0.35
	elif current_stamina <= low_threshold:
		rate *= 1.25
	current_stamina = minf(max_stamina, current_stamina + rate * delta)
	stamina_changed.emit(current_stamina, max_stamina)


func spend(amount: float) -> bool:
	if not GameState.is_match_active():
		return false
	if current_stamina < min_to_act:
		return false
	current_stamina = maxf(0.0, current_stamina - amount)
	_regen_delay = REGEN_DELAY_AFTER_USE
	stamina_changed.emit(current_stamina, max_stamina)
	return true


func drain(amount: float, from_hit: bool = false) -> void:
	if not GameState.is_match_active():
		return
	if amount <= 0.0:
		return
	current_stamina = maxf(0.0, current_stamina - amount)
	_regen_delay = REGEN_DELAY_AFTER_HIT if from_hit else REGEN_DELAY_AFTER_USE
	stamina_changed.emit(current_stamina, max_stamina)


func is_low() -> bool:
	return current_stamina <= low_threshold


func is_exhausted() -> bool:
	return current_stamina <= low_threshold * 0.35

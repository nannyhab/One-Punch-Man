class_name HealthComponent
extends Node

signal health_changed(current: int, maximum: int)
signal knocked_out

@export var max_health: int = 100

var current_health: int = 100


func _ready() -> void:
	current_health = max_health


func take_damage(amount: int) -> void:
	if not GameState.is_match_active():
		return
	if amount <= 0:
		return
	current_health = maxi(0, current_health - amount)
	health_changed.emit(current_health, max_health)
	if current_health <= 0:
		knocked_out.emit()

extends Camera2D
class_name CameraShake

var shake_strength: float = 0.0
var shake_decay: float = 8.0
var _rng := RandomNumberGenerator.new()


func _process(delta: float) -> void:
	if shake_strength > 0.0:
		offset = Vector2(
			_rng.randf_range(-shake_strength, shake_strength),
			_rng.randf_range(-shake_strength, shake_strength)
		)
		shake_strength = move_toward(shake_strength, 0.0, shake_decay * delta)
	else:
		offset = Vector2.ZERO


func shake(amount: float = 4.0) -> void:
	shake_strength = maxf(shake_strength, amount)

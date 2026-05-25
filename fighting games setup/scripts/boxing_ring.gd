extends Node2D
class_name BoxingRing

@export var ring_rect: Rect2 = Rect2(-60, 50, 1400, 630)

signal bounds_ready(bounds: Rect2)


func _ready() -> void:
	bounds_ready.emit(ring_rect)


func clamp_position(pos: Vector2, margin: float = 40.0) -> Vector2:
	return Vector2(
		clampf(pos.x, ring_rect.position.x + margin, ring_rect.end.x - margin),
		clampf(pos.y, ring_rect.position.y + margin, ring_rect.end.y - margin)
	)


func get_center() -> Vector2:
	return ring_rect.get_center()

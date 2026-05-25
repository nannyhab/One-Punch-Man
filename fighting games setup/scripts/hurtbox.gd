class_name Hurtbox
extends Area2D

enum HurtType { HEAD, BODY, GUARD }

@export var hurt_type: HurtType = HurtType.BODY

var owner_boxer: Boxer = null

const DEBUG_HITBOXES := false


func _ready() -> void:
	monitorable = true
	monitoring = false


func setup(boxer: Boxer, is_player: bool) -> void:
	owner_boxer = boxer
	collision_layer = 1 if is_player else 2
	collision_mask = 0
	queue_redraw()


func _draw() -> void:
	if not DEBUG_HITBOXES:
		return
	var shape_node := $CollisionShape2D as CollisionShape2D
	if shape_node == null:
		return
	if shape_node.shape is RectangleShape2D:
		var size := (shape_node.shape as RectangleShape2D).size
		draw_rect(Rect2(-size * 0.5, size), Color(1.0, 0.1, 0.1, 0.35), false, 2.0)

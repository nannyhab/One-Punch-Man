class_name Hitbox
extends Area2D

signal hit_connected(hurtbox: Hurtbox)

enum PunchKind { STRAIGHT, HOOK, UPPERCUT }

@export var punch_kind: PunchKind = PunchKind.STRAIGHT

var owner_boxer: Boxer = null
var active: bool = false
var has_connected: bool = false

const DEBUG_HITBOXES := false


func _ready() -> void:
	monitoring = false
	monitorable = false
	area_entered.connect(_on_area_entered)
	collision_layer = 0
	collision_mask = 0


func setup(boxer: Boxer, is_player: bool) -> void:
	owner_boxer = boxer
	collision_layer = 4 if is_player else 8
	collision_mask = 2 if is_player else 1


func begin_punch_window() -> void:
	active = true
	has_connected = false
	monitoring = true
	queue_redraw()
	_check_overlaps()


func refresh_overlaps() -> void:
	if active and not has_connected:
		_check_overlaps()


func end_punch_window() -> void:
	active = false
	monitoring = false
	queue_redraw()


func _check_overlaps() -> void:
	for area in get_overlapping_areas():
		_on_area_entered(area)


func _on_area_entered(area: Area2D) -> void:
	if not active or has_connected:
		return
	if area is Hurtbox:
		has_connected = true
		hit_connected.emit(area)


func update_position_and_shape(offset: Vector2, size: Vector2, facing: int) -> void:
	var shape_node := $CollisionShape2D as CollisionShape2D
	if shape_node and shape_node.shape is RectangleShape2D:
		(shape_node.shape as RectangleShape2D).size = size
	position = Vector2(offset.x * facing, offset.y)
	queue_redraw()


func _draw() -> void:
	if not DEBUG_HITBOXES:
		return
	var shape_node := $CollisionShape2D as CollisionShape2D
	if shape_node == null or not shape_node.shape is RectangleShape2D:
		return
	var size := (shape_node.shape as RectangleShape2D).size
	var color := Color.YELLOW if has_connected else Color(0.0, 1.0, 0.25, 0.35)
	draw_rect(Rect2(-size * 0.5, size), color, false, 2.0)

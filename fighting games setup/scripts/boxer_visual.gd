extends Node2D

@export var glove_color: Color = Color(0.29, 0.56, 0.85)
@export var shorts_color: Color = Color(0.18, 0.35, 0.65)
@export var skin_color: Color = Color(0.85, 0.65, 0.52)
@export var hair_color: Color = Color(0.12, 0.1, 0.14)

const BODY_SCALE := 1.42
const OUTLINE := Color(0.025, 0.022, 0.035, 0.92)
const MAX_PART_DISTANCE := 175.0

const LEAD_SHOULDER := Vector2(23, -60)
const REAR_SHOULDER := Vector2(-21, -58)
const LEAD_GUARD := Vector2(36, -50)
const REAR_GUARD := Vector2(-12, -51)

@onready var anim: AnimationPlayer = $AnimationPlayer
@onready var shadow: Polygon2D = $Shadow
@onready var outline: Polygon2D = $Outline

var body: Node2D
var punch_target_x := 92.0
var punch_target_y := -50.0


func _ready() -> void:
	scale = Vector2(BODY_SCALE, BODY_SCALE)
	_hide_template_parts()
	_build_body()
	_build_animations()
	_apply_guard_pose()
	_play("idle_guard")
	set_process(true)


func _process(_delta: float) -> void:
	_validate_pose()


func aim_next_punch(punch_type: int, target_local: Vector2, _facing_dir: int) -> void:
	match punch_type:
		PunchData.Type.STRAIGHT:
			punch_target_x = clampf(target_local.x, 118.0, 135.0)
			punch_target_y = -50.0
			_replace_animation("straight", _make_straight())
		PunchData.Type.HOOK:
			punch_target_x = clampf(target_local.x, 98.0, 115.0)
			punch_target_y = -41.0
			_replace_animation("hook", _make_hook())
		PunchData.Type.UPPERCUT:
			punch_target_x = clampf(target_local.x, 82.0, 95.0)
			punch_target_y = -74.0
			_replace_animation("uppercut", _make_uppercut())


func get_punch_hitbox_offset(punch_type: int) -> Vector2:
	var local_target := Vector2(punch_target_x, punch_target_y)
	var body_offset := Vector2.ZERO
	match punch_type:
		PunchData.Type.STRAIGHT:
			local_target = Vector2(punch_target_x, -50.0)
			body_offset = Vector2(8.0, 0.0)
		PunchData.Type.HOOK:
			local_target = Vector2(punch_target_x, -41.0)
			body_offset = Vector2(7.0, 0.0)
		PunchData.Type.UPPERCUT:
			local_target = Vector2(punch_target_x, -74.0)
			body_offset = Vector2(5.0, -3.0)
	var visual_scale := absf(scale.x)
	if visual_scale <= 0.0:
		visual_scale = BODY_SCALE
	return (local_target + body_offset) * visual_scale


func _hide_template_parts() -> void:
	for part_name in [
		"Outline", "LegBack", "ShoeBack", "LegFront", "ShoeFront", "Shorts", "ShortsStripe",
		"Torso", "Head", "Hair", "ArmRear", "GloveRear", "ArmLead", "GloveLead",
	]:
		var node := get_node_or_null(part_name)
		if node and node != outline:
			node.visible = false
	if outline:
		outline.visible = false


func _build_body() -> void:
	body = Node2D.new()
	body.name = "Body"
	add_child(body)
	move_child(body, 0)

	shadow.polygon = PackedVector2Array([
		Vector2(-46, 54), Vector2(46, 54), Vector2(36, 66), Vector2(-36, 66),
	])
	shadow.color = Color(0, 0, 0, 0.38)
	shadow.z_index = -20

	_part("BackLeg", _limb_shape(48, 9, 7), skin_color.darkened(0.18), -8)
	_part("BackShoe", _shoe_shape(), Color(0.08, 0.08, 0.12), -7)
	_part("FrontLeg", _limb_shape(50, 9, 7), skin_color.darkened(0.06), 4)
	_part("FrontShoe", _shoe_shape(), Color(0.08, 0.08, 0.12), 5)
	_part("Shorts", PackedVector2Array([
		Vector2(-29, -7), Vector2(29, -7), Vector2(31, 20),
		Vector2(10, 31), Vector2(0, 23), Vector2(-10, 31), Vector2(-31, 20),
	]), shorts_color, 8)
	_part("Torso", PackedVector2Array([
		Vector2(-20, -64), Vector2(20, -64), Vector2(38, -38), Vector2(30, -2),
		Vector2(15, 21), Vector2(-15, 21), Vector2(-30, -2), Vector2(-38, -38),
	]), skin_color.darkened(0.08), 10)
	_part("ChestLight", PackedVector2Array([
		Vector2(-22, -44), Vector2(-4, -50), Vector2(-2, -27), Vector2(-19, -25),
		Vector2(4, -50), Vector2(23, -44), Vector2(18, -25), Vector2(2, -27),
	]), skin_color.lightened(0.06), 11, false)
	_part("Neck", PackedVector2Array([
		Vector2(-7, -74), Vector2(7, -74), Vector2(9, -58), Vector2(-9, -58),
	]), skin_color.darkened(0.05), 12)
	_part("Head", _head_shape(), skin_color, 20)
	_part("Hair", PackedVector2Array([
		Vector2(-13, -17), Vector2(13, -17), Vector2(13, -7), Vector2(5, -3), Vector2(-12, -6),
	]), hair_color, 21, false)
	_part("Eye", PackedVector2Array([
		Vector2(2, -4), Vector2(9, -4), Vector2(9, -1), Vector2(2, -1),
	]), Color(0.03, 0.025, 0.02), 22, false)
	_part("RearArm", _arm_shape(44), skin_color.darkened(0.11), 13)
	_part("RearGlove", _glove_shape(), glove_color, 16)
	_part("LeadArm", _arm_shape(48), skin_color.darkened(0.03), 24)
	_part("LeadGlove", _glove_shape(), glove_color, 28)
	_ensure_required_parts_visible()


func _part(part_name: String, points: PackedVector2Array, color: Color, z: int, with_outline: bool = true) -> Polygon2D:
	var part := Polygon2D.new()
	part.name = part_name
	part.polygon = points
	part.color = color
	part.z_index = z
	body.add_child(part)
	if with_outline:
		var outline_part := Polygon2D.new()
		outline_part.name = part_name + "Outline"
		outline_part.polygon = _scaled_polygon(points, 1.12)
		outline_part.color = OUTLINE
		outline_part.z_index = -1
		part.add_child(outline_part)
		part.move_child(outline_part, 0)
	return part


func _scaled_polygon(points: PackedVector2Array, amount: float) -> PackedVector2Array:
	var scaled := PackedVector2Array()
	for point in points:
		scaled.append(point * amount)
	return scaled


func _limb_shape(length: float, start_width: float, end_width: float) -> PackedVector2Array:
	return PackedVector2Array([
		Vector2(0, -start_width), Vector2(length, -end_width),
		Vector2(length + 2, end_width), Vector2(0, start_width),
	])


func _arm_shape(length: float) -> PackedVector2Array:
	return PackedVector2Array([
		Vector2(0, -7), Vector2(length, -6), Vector2(length + 2, 6), Vector2(0, 8),
	])


func _shoe_shape() -> PackedVector2Array:
	return PackedVector2Array([
		Vector2(-5, -6), Vector2(25, -8), Vector2(35, -1), Vector2(28, 8), Vector2(-7, 8),
	])


func _head_shape() -> PackedVector2Array:
	return PackedVector2Array([
		Vector2(-11, -17), Vector2(11, -17), Vector2(16, -6), Vector2(14, 8),
		Vector2(6, 17), Vector2(-6, 17), Vector2(-14, 8), Vector2(-16, -6),
	])


func _glove_shape() -> PackedVector2Array:
	return PackedVector2Array([
		Vector2(-17, -15), Vector2(17, -15), Vector2(23, -5), Vector2(21, 10),
		Vector2(10, 18), Vector2(-15, 16), Vector2(-22, 3), Vector2(-21, -8),
	])


func _build_animations() -> void:
	var lib := AnimationLibrary.new()
	lib.add_animation("idle_guard", _make_idle())
	lib.add_animation("guard", _make_guard())
	lib.add_animation("move_forward", _make_step(1.0))
	lib.add_animation("move_backward", _make_step(-1.0))
	lib.add_animation("dodge_left", _make_dodge(-24))
	lib.add_animation("dodge_right", _make_dodge(24))
	lib.add_animation("straight", _make_straight())
	lib.add_animation("hook", _make_hook())
	lib.add_animation("uppercut", _make_uppercut())
	lib.add_animation("hit_head", _make_hit(true))
	lib.add_animation("hit_body", _make_hit(false))
	lib.add_animation("block_react", _make_block())
	lib.add_animation("stunned", _make_stunned())
	lib.add_animation("knockdown", _make_knockdown())
	anim.add_animation_library("", lib)
	anim.root_node = NodePath("..")


func _replace_animation(anim_name: String, replacement: Animation) -> void:
	var lib := anim.get_animation_library("")
	if lib == null:
		return
	if lib.has_animation(anim_name):
		lib.remove_animation(anim_name)
	lib.add_animation(anim_name, replacement)


func _make_idle() -> Animation:
	var a := Animation.new()
	a.length = 0.9
	a.loop_mode = Animation.LOOP_LINEAR
	_pose_keys(a, 0.0, _guard_pose())
	_key(a, "Body:position:y", 0.0, 0.0)
	_key(a, "Body:position:y", 0.45, -3.0)
	_key(a, "Body:position:y", 0.9, 0.0)
	_pose_keys(a, 0.9, _guard_pose())
	return a


func _make_guard() -> Animation:
	var a := Animation.new()
	a.length = 0.22
	_pose_keys(a, 0.0, _guard_pose())
	return a


func _make_step(dir: float) -> Animation:
	var a := Animation.new()
	a.length = 0.32
	a.loop_mode = Animation.LOOP_LINEAR
	_pose_keys(a, 0.0, _guard_pose())
	_key(a, "Body:position:x", 0.0, 0.0)
	_key(a, "Body:position:x", 0.16, dir * 4.0)
	_key(a, "Body:position:x", 0.32, 0.0)
	_pose_keys(a, 0.32, _guard_pose())
	return a


func _make_dodge(offset: float) -> Animation:
	var a := Animation.new()
	a.length = 0.22
	_pose_keys(a, 0.0, _guard_pose())
	_key(a, "Body:position:x", 0.0, 0.0)
	_key(a, "Body:position:x", 0.08, offset)
	_key(a, "Body:position:x", 0.22, 0.0)
	return a


func _make_straight() -> Animation:
	var a := Animation.new()
	a.length = 0.42
	_pose_keys(a, 0.0, _guard_pose())
	var active := _guard_pose()
	_set_arm_pose(active, "Lead", LEAD_SHOULDER, Vector2(punch_target_x, -50))
	_set_arm_pose(active, "Rear", REAR_SHOULDER, REAR_GUARD)
	_pose_keys(a, 0.13, active)
	_key(a, "Body:position:x", 0.13, 8.0)
	_key(a, "Body:rotation", 0.13, deg_to_rad(2))
	_pose_keys(a, 0.42, _guard_pose())
	_key(a, "Body:position:x", 0.42, 0.0)
	_key(a, "Body:rotation", 0.42, 0.0)
	return a


func _make_hook() -> Animation:
	var a := Animation.new()
	a.length = 0.58
	_pose_keys(a, 0.0, _guard_pose())
	var startup := _guard_pose()
	_set_arm_pose(startup, "Lead", LEAD_SHOULDER, Vector2(22, -42))
	_pose_keys(a, 0.12, startup)
	var active := _guard_pose()
	_set_arm_pose(active, "Lead", LEAD_SHOULDER, Vector2(punch_target_x, -41))
	_pose_keys(a, 0.30, active)
	_key(a, "Body:rotation", 0.30, deg_to_rad(5))
	_pose_keys(a, 0.58, _guard_pose())
	_key(a, "Body:rotation", 0.58, 0.0)
	return a


func _make_uppercut() -> Animation:
	var a := Animation.new()
	a.length = 0.72
	_pose_keys(a, 0.0, _guard_pose())
	var dip := _guard_pose()
	_set_arm_pose(dip, "Lead", LEAD_SHOULDER, Vector2(28, -18))
	_pose_keys(a, 0.16, dip)
	_key(a, "Body:position:y", 0.16, 7.0)
	var active := _guard_pose()
	_set_arm_pose(active, "Lead", LEAD_SHOULDER, Vector2(punch_target_x, -74))
	_pose_keys(a, 0.34, active)
	_key(a, "Body:position:y", 0.34, -3.0)
	_pose_keys(a, 0.72, _guard_pose())
	_key(a, "Body:position:y", 0.72, 0.0)
	return a


func _make_hit(is_head: bool) -> Animation:
	var a := Animation.new()
	a.length = 0.32
	_pose_keys(a, 0.0, _guard_pose())
	_key(a, "Body:position:x", 0.0, 0.0)
	_key(a, "Body:position:x", 0.06, -10.0 if is_head else -6.0)
	_key(a, "Body:position:x", 0.32, 0.0)
	return a


func _make_block() -> Animation:
	var a := Animation.new()
	a.length = 0.24
	var block_pose := _guard_pose()
	_set_arm_pose(block_pose, "Lead", LEAD_SHOULDER, Vector2(28, -52))
	_set_arm_pose(block_pose, "Rear", REAR_SHOULDER, Vector2(-4, -53))
	_pose_keys(a, 0.0, block_pose)
	_key(a, "Body:position:x", 0.05, -6.0)
	_key(a, "Body:position:x", 0.24, 0.0)
	return a


func _make_stunned() -> Animation:
	var a := Animation.new()
	a.length = 1.0
	a.loop_mode = Animation.LOOP_LINEAR
	_pose_keys(a, 0.0, _guard_pose())
	_key(a, "Body:rotation", 0.0, deg_to_rad(-3))
	_key(a, "Body:rotation", 0.5, deg_to_rad(3))
	_key(a, "Body:rotation", 1.0, deg_to_rad(-3))
	return a


func _make_knockdown() -> Animation:
	var a := Animation.new()
	a.length = 0.8
	a.loop_mode = Animation.LOOP_NONE
	_pose_keys(a, 0.0, _guard_pose())
	_key(a, "Body:rotation", 0.0, 0.0)
	_key(a, "Body:rotation", 0.35, deg_to_rad(38))
	_key(a, "Body:position:x", 0.35, -40.0)
	_key(a, "Body:position:y", 0.35, 26.0)
	return a


func _guard_pose() -> Dictionary:
	var pose := {}
	pose["Body"] = {"position": Vector2.ZERO, "rotation": 0.0}
	pose["BackLeg"] = {"position": Vector2(-17, 19), "rotation": deg_to_rad(104), "scale": Vector2.ONE}
	pose["BackShoe"] = {"position": Vector2(-28, 82), "rotation": deg_to_rad(-5), "scale": Vector2.ONE}
	pose["FrontLeg"] = {"position": Vector2(16, 17), "rotation": deg_to_rad(73), "scale": Vector2.ONE}
	pose["FrontShoe"] = {"position": Vector2(24, 72), "rotation": deg_to_rad(3), "scale": Vector2.ONE}
	pose["Torso"] = {"position": Vector2(0, -25), "rotation": deg_to_rad(-2)}
	pose["ChestLight"] = {"position": Vector2(0, -25), "rotation": deg_to_rad(-2)}
	pose["Shorts"] = {"position": Vector2(0, 4), "rotation": 0.0, "scale": Vector2.ONE}
	pose["Neck"] = {"position": Vector2(0, -1), "rotation": 0.0, "scale": Vector2.ONE}
	pose["Head"] = {"position": Vector2(0, -84), "rotation": deg_to_rad(-2), "scale": Vector2.ONE}
	pose["Hair"] = {"position": Vector2(0, -84), "rotation": deg_to_rad(-2), "scale": Vector2.ONE}
	pose["Eye"] = {"position": Vector2(0, -84), "rotation": deg_to_rad(-2), "scale": Vector2.ONE}
	_set_arm_pose(pose, "Lead", LEAD_SHOULDER, LEAD_GUARD)
	_set_arm_pose(pose, "Rear", REAR_SHOULDER, REAR_GUARD)
	return pose


func _set_arm_pose(pose: Dictionary, prefix: String, shoulder: Vector2, glove: Vector2) -> void:
	var vector := glove - shoulder
	if vector.length() < 1.0:
		vector = Vector2(1, 0)
	var midpoint := shoulder + vector * 0.5
	pose[prefix + "Arm"] = {
		"position": midpoint,
		"rotation": vector.angle(),
		"scale": Vector2(clampf(vector.length() / 48.0, 0.35, 3.0), 1.0),
	}
	pose[prefix + "Glove"] = {"position": glove, "rotation": vector.angle()}


func _pose_keys(a: Animation, time: float, pose: Dictionary) -> void:
	for part_name in pose.keys():
		var values: Dictionary = pose[part_name]
		var base_path := "Body" if part_name == "Body" else "Body/" + str(part_name)
		if values.has("position"):
			_key(a, base_path + ":position", time, values["position"])
		if values.has("rotation"):
			_key(a, base_path + ":rotation", time, values["rotation"])
		if values.has("scale"):
			_key(a, base_path + ":scale", time, values["scale"])


func _key(a: Animation, path: String, time: float, value: Variant) -> void:
	var node_path := NodePath(path)
	var track := -1
	for i in range(a.get_track_count()):
		if a.track_get_path(i) == node_path:
			track = i
			break
	if track == -1:
		track = a.add_track(Animation.TYPE_VALUE)
		a.track_set_path(track, node_path)
	a.track_insert_key(track, time, value)


func _apply_guard_pose() -> void:
	var pose := _guard_pose()
	for part_name in pose.keys():
		var node := body if part_name == "Body" else body.get_node_or_null(str(part_name))
		if node == null:
			continue
		node.visible = true
		var values: Dictionary = pose[part_name]
		if values.has("position"):
			node.position = values["position"]
		if values.has("rotation"):
			node.rotation = values["rotation"]
		if values.has("scale"):
			node.scale = values["scale"]
	_ensure_required_parts_visible()


func _validate_pose() -> void:
	if body == null:
		return
	if not _ensure_required_parts_visible():
		_apply_guard_pose()
		if anim:
			anim.play("guard")
		print("Invalid pose detected: resetting fighter")
		return
	for child in body.get_children():
		if child is Node2D and (child as Node2D).position.length() > MAX_PART_DISTANCE:
			_apply_guard_pose()
			if anim:
				anim.play("guard")
			print("Invalid pose detected: resetting fighter")
			return


func _ensure_required_parts_visible() -> bool:
	if body == null:
		return false
	var ok := true
	for part_name in [
		"Head", "Torso", "Shorts", "LeadArm", "RearArm", "LeadGlove", "RearGlove",
		"FrontLeg", "BackLeg", "FrontShoe", "BackShoe",
	]:
		var node := body.get_node_or_null(part_name)
		if node == null:
			ok = false
		elif node is CanvasItem:
			(node as CanvasItem).visible = true
	return ok


func _play(anim_name: String) -> void:
	if anim and anim.has_animation(anim_name):
		anim.play(anim_name)

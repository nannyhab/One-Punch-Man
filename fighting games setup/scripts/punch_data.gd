class_name PunchData
extends Resource

enum Type { STRAIGHT, HOOK, UPPERCUT }

@export var punch_type: Type = Type.STRAIGHT
@export var display_name: String = "Straight"
@export var startup: float = 0.10
@export var active: float = 0.08
@export var recovery: float = 0.22
@export var damage: int = 8
@export var stamina_cost: int = 10
@export var hitbox_size: Vector2 = Vector2(30, 24)
@export var hitbox_offset: Vector2 = Vector2(55, -30)
@export var counter_bonus_damage: int = 5


static func create_defaults() -> Dictionary:
	return {
		Type.STRAIGHT: _make(
			Type.STRAIGHT, "Straight", 0.10, 0.08, 0.22,
			8, 10, Vector2(100, 34), Vector2(145, -50)
		),
		Type.HOOK: _make(
			Type.HOOK, "Hook", 0.16, 0.10, 0.32,
			13, 10, Vector2(90, 50), Vector2(125, -48)
		),
		Type.UPPERCUT: _make(
			Type.UPPERCUT, "Uppercut", 0.22, 0.10, 0.42,
			18, 10, Vector2(110, 70), Vector2(105, -52)
		),
	}


static func _make(
	ptype: Type, pname: String, p_startup: float, p_active: float, p_recovery: float,
	p_damage: int, p_stamina: int, hsize: Vector2, hoffset: Vector2
) -> PunchData:
	var p := PunchData.new()
	p.punch_type = ptype
	p.display_name = pname
	p.startup = p_startup
	p.active = p_active
	p.recovery = p_recovery
	p.damage = p_damage
	p.stamina_cost = p_stamina
	p.hitbox_size = hsize
	p.hitbox_offset = hoffset
	return p

extends Control

@onready var panel: PanelContainer = $Panel
@onready var title_label: Label = $Panel/VBox/Title
@onready var subtitle_label: Label = $Panel/VBox/Subtitle
@onready var player1_input: LineEdit = $Panel/VBox/Player1Input
@onready var player2_input: LineEdit = $Panel/VBox/Player2Input
@onready var player1_label: Label = $Panel/VBox/Player1NameLabel
@onready var player2_label: Label = $Panel/VBox/Player2NameLabel
@onready var vs_cpu_button: Button = $Panel/VBox/VsCpuButton
@onready var vs_player_button: Button = $Panel/VBox/VsPlayerButton
@onready var hint_label: Label = $Panel/VBox/Hint

var _time: float = 0.0
var _dust: Array[Vector2] = []

const BLUE := Color(0.16, 0.48, 1.0)
const RED := Color(1.0, 0.18, 0.22)
const CARD_BG := Color(0.035, 0.04, 0.07, 0.86)
const CARD_BORDER := Color(0.58, 0.62, 0.74, 0.34)


func _ready() -> void:
	GameState.match_state = GameState.MatchState.MENU
	_build_dust()
	_apply_menu_theme()
	_style_button(vs_cpu_button, BLUE)
	_style_button(vs_player_button, RED)
	_connect_button_fx(vs_cpu_button)
	_connect_button_fx(vs_player_button)
	vs_cpu_button.pressed.connect(_on_vs_cpu)
	vs_player_button.pressed.connect(_on_vs_player)
	player1_input.text = GameState.player1_name
	player2_input.text = GameState.player2_name
	player1_input.grab_focus()
	set_process(true)


func _process(delta: float) -> void:
	_time += delta
	title_label.scale = Vector2.ONE * (1.0 + sin(_time * 2.0) * 0.015)
	title_label.pivot_offset = title_label.size * 0.5
	queue_redraw()


func _draw() -> void:
	var rect := Rect2(Vector2.ZERO, size)
	_draw_arena_background(rect)
	_draw_spotlights(rect)
	_draw_ring_silhouette(rect)
	_draw_crowd(rect)
	_draw_dust(rect)


func _draw_arena_background(rect: Rect2) -> void:
	draw_rect(rect, Color(0.018, 0.02, 0.036))
	for i in range(8):
		var t := float(i) / 7.0
		var band := Rect2(0, rect.size.y * t, rect.size.x, rect.size.y / 7.0 + 2.0)
		draw_rect(band, Color(0.025 + t * 0.02, 0.028 + t * 0.012, 0.055 + t * 0.03, 1.0))


func _draw_spotlights(rect: Rect2) -> void:
	var center := Vector2(rect.size.x * 0.5 + sin(_time * 0.45) * 22.0, rect.size.y * 0.22)
	for i in range(9, 0, -1):
		var alpha := 0.018 * float(10 - i)
		draw_circle(center, 44.0 * float(i), Color(0.9, 0.92, 1.0, alpha))
	draw_line(Vector2(rect.size.x * 0.18, 0), center + Vector2(-80, 170), Color(0.25, 0.32, 0.55, 0.11), 70.0)
	draw_line(Vector2(rect.size.x * 0.82, 0), center + Vector2(80, 170), Color(0.55, 0.25, 0.28, 0.10), 70.0)


func _draw_ring_silhouette(rect: Rect2) -> void:
	var y := rect.size.y * 0.66
	var left := rect.size.x * 0.13
	var right := rect.size.x * 0.87
	var rope_color := Color(0.78, 0.82, 0.9, 0.22)
	for offset in [-42.0, -18.0, 8.0]:
		draw_line(Vector2(left, y + offset), Vector2(right, y + offset), rope_color, 4.0)
	for x in [left, right]:
		draw_rect(Rect2(x - 5, y - 68, 10, 100), Color(0.72, 0.74, 0.82, 0.18))
	draw_colored_polygon(
		PackedVector2Array([
			Vector2(left + 42, y + 36), Vector2(right - 42, y + 36),
			Vector2(right - 92, y + 72), Vector2(left + 92, y + 72),
		]),
		Color(0, 0, 0, 0.34)
	)


func _draw_crowd(rect: Rect2) -> void:
	var base_y := rect.size.y * 0.18
	for i in range(18):
		var x := 45.0 + i * 70.0
		var h := 18.0 + float((i * 17) % 28)
		draw_colored_polygon(
			PackedVector2Array([
				Vector2(x - 18, base_y + h), Vector2(x, base_y), Vector2(x + 18, base_y + h),
				Vector2(x + 12, base_y + h + 34), Vector2(x - 12, base_y + h + 34),
			]),
			Color(0.005, 0.007, 0.014, 0.42)
		)


func _draw_dust(_rect: Rect2) -> void:
	for i in range(_dust.size()):
		var p := _dust[i]
		var drift := Vector2(sin(_time * 0.6 + float(i)) * 8.0, fmod(_time * (4.0 + i % 3), 26.0))
		draw_circle(p + drift, 1.4, Color(1.0, 0.9, 0.65, 0.16))


func _build_dust() -> void:
	_dust.clear()
	for i in range(28):
		_dust.append(Vector2(80 + (i * 43) % 1120, 130 + (i * 67) % 470))


func _apply_menu_theme() -> void:
	panel.add_theme_stylebox_override("panel", _card_style())
	panel.add_theme_constant_override("content_margin_left", 34)
	panel.add_theme_constant_override("content_margin_right", 34)
	panel.add_theme_constant_override("content_margin_top", 28)
	panel.add_theme_constant_override("content_margin_bottom", 28)

	title_label.add_theme_color_override("font_color", Color(0.96, 0.97, 1.0))
	title_label.add_theme_color_override("font_shadow_color", Color(0.0, 0.0, 0.0, 0.75))
	title_label.add_theme_constant_override("shadow_offset_x", 3)
	title_label.add_theme_constant_override("shadow_offset_y", 4)
	subtitle_label.add_theme_color_override("font_color", Color(0.86, 0.88, 0.96, 0.9))
	player1_label.add_theme_color_override("font_color", BLUE.lightened(0.35))
	player2_label.add_theme_color_override("font_color", RED.lightened(0.28))
	hint_label.add_theme_color_override("font_color", Color(0.62, 0.65, 0.74, 0.72))

	_style_input(player1_input, BLUE)
	_style_input(player2_input, RED)


func _card_style() -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = CARD_BG
	style.border_color = CARD_BORDER
	style.set_border_width_all(2)
	style.set_corner_radius_all(22)
	style.shadow_color = Color(0, 0, 0, 0.48)
	style.shadow_size = 18
	style.shadow_offset = Vector2(0, 8)
	return style


func _style_input(input: LineEdit, accent: Color) -> void:
	input.add_theme_stylebox_override("normal", _field_style(Color(0.08, 0.095, 0.14, 0.95), Color(0.22, 0.25, 0.35, 0.8)))
	input.add_theme_stylebox_override("focus", _field_style(Color(0.09, 0.105, 0.16, 0.98), accent.lightened(0.18)))
	input.add_theme_color_override("font_color", Color(0.96, 0.97, 1.0))
	input.add_theme_color_override("font_placeholder_color", Color(0.62, 0.66, 0.76, 0.72))
	input.add_theme_color_override("caret_color", accent.lightened(0.35))
	input.add_theme_color_override("selection_color", accent.darkened(0.15))
	input.add_theme_font_size_override("font_size", 16)


func _field_style(bg: Color, border: Color) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = bg
	style.border_color = border
	style.set_border_width_all(2)
	style.set_corner_radius_all(14)
	style.content_margin_left = 16
	style.content_margin_right = 16
	return style


func _style_button(button: Button, accent: Color) -> void:
	button.add_theme_stylebox_override("normal", _button_style(accent, 0.0))
	button.add_theme_stylebox_override("hover", _button_style(accent.lightened(0.16), 0.18))
	button.add_theme_stylebox_override("pressed", _button_style(accent.darkened(0.12), 0.08))
	button.add_theme_color_override("font_color", Color.WHITE)
	button.add_theme_color_override("font_hover_color", Color.WHITE)
	button.add_theme_color_override("font_pressed_color", Color(0.9, 0.92, 1.0))
	button.add_theme_font_size_override("font_size", 18)


func _button_style(accent: Color, glow: float) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = accent.darkened(0.35)
	style.border_color = accent.lightened(0.15)
	style.set_border_width_all(2)
	style.set_corner_radius_all(16)
	style.shadow_color = Color(accent.r, accent.g, accent.b, 0.22 + glow)
	style.shadow_size = 8 + int(glow * 22.0)
	style.shadow_offset = Vector2.ZERO
	return style


func _connect_button_fx(button: Button) -> void:
	button.mouse_entered.connect(_animate_button.bind(button, Vector2(1.035, 1.035)))
	button.mouse_exited.connect(_animate_button.bind(button, Vector2.ONE))
	button.button_down.connect(_animate_button.bind(button, Vector2(0.985, 0.965)))
	button.button_up.connect(_animate_button.bind(button, Vector2(1.035, 1.035)))
	button.resized.connect(_set_button_pivot.bind(button))
	_set_button_pivot(button)


func _set_button_pivot(button: Button) -> void:
	button.pivot_offset = button.size * 0.5


func _animate_button(button: Button, target_scale: Vector2) -> void:
	var tween := create_tween()
	tween.tween_property(button, "scale", target_scale, 0.08).set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_OUT)


func _on_vs_cpu() -> void:
	GameState.mode = GameState.Mode.VS_CPU
	_save_names()
	get_tree().change_scene_to_file("res://scenes/Main.tscn")


func _on_vs_player() -> void:
	GameState.mode = GameState.Mode.VS_PLAYER
	_save_names()
	get_tree().change_scene_to_file("res://scenes/Main.tscn")


func _save_names() -> void:
	GameState.player1_name = _clean_name(player1_input.text, "Player 1")
	GameState.player2_name = _clean_name(player2_input.text, "Player 2")


func _clean_name(raw_name: String, fallback: String) -> String:
	var cleaned := raw_name.strip_edges()
	if cleaned.is_empty():
		return fallback
	return cleaned.left(14)

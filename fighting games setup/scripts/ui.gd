extends CanvasLayer

signal restart_requested
signal menu_requested

@onready var right_label: Label = $HUD/TopBar/CPURow/CPULabel
@onready var player_label: Label = $HUD/TopBar/PlayerRow/PlayerLabel
@onready var player_health_bar: ProgressBar = $HUD/TopBar/PlayerRow/HealthBar
@onready var right_health_bar: ProgressBar = $HUD/TopBar/CPURow/HealthBar
@onready var player_stamina_bar: ProgressBar = $HUD/TopBar/PlayerRow/StaminaBar
@onready var right_stamina_bar: ProgressBar = $HUD/TopBar/CPURow/StaminaBar
@onready var timer_label: Label = $HUD/TopBar/TimerLabel
@onready var score_label: Label = $HUD/ScoreLabel
@onready var feedback_label: Label = $HUD/FeedbackLabel
@onready var controls_hint: Label = $HUD/ControlsHint
@onready var debug_distance_label: Label = $HUD/DebugDistanceLabel
@onready var result_panel: PanelContainer = $HUD/ResultPanel
@onready var result_label: Label = $HUD/ResultPanel/VBox/ResultLabel
@onready var restart_button: Button = $HUD/ResultPanel/VBox/RestartButton
@onready var menu_button: Button = $HUD/ResultPanel/VBox/MenuButton

var feedback_timer: float = 0.0
var _vs_cpu: bool = true
var _left_stamina_target: float = 100.0
var _right_stamina_target: float = 100.0
var _left_stamina_display: float = 100.0
var _right_stamina_display: float = 100.0
var _left_boxer: Boxer = null
var _right_boxer: Boxer = null
var _left_move_label: Label
var _right_move_label: Label
var _left_move_timer: float = 0.0
var _right_move_timer: float = 0.0
var _left_name: String = "Player 1"
var _right_name: String = "IRON MIKE AI"

const STAMINA_SMOOTH_SPEED := 7.0
const MOVE_LABEL_TIME := 0.55
const DEBUG_SHOW_SCORE := false
const STAMINA_NORMAL_COLOR := Color(0.35, 0.75, 1.0)
const STAMINA_LOW_COLOR := Color(1.0, 0.72, 0.2)
const HEALTH_COLOR := Color(0.25, 0.9, 0.35)


func _ready() -> void:
	_left_move_label = _make_move_label(Color(0.45, 0.75, 1.0))
	_right_move_label = _make_move_label(Color(1.0, 0.45, 0.45))
	$HUD.add_child(_left_move_label)
	$HUD.add_child(_right_move_label)
	score_label.visible = DEBUG_SHOW_SCORE
	debug_distance_label.visible = false
	result_panel.visible = false
	_style_bar(player_health_bar, HEALTH_COLOR)
	_style_bar(right_health_bar, HEALTH_COLOR)
	_style_bar(player_stamina_bar, STAMINA_NORMAL_COLOR)
	_style_bar(right_stamina_bar, STAMINA_NORMAL_COLOR)
	restart_button.pressed.connect(_on_restart_pressed)
	menu_button.pressed.connect(_on_menu_pressed)


func _process(delta: float) -> void:
	if feedback_timer > 0.0:
		feedback_timer -= delta
		if feedback_timer <= 0.0:
			feedback_label.text = ""

	_left_stamina_display = lerpf(_left_stamina_display, _left_stamina_target, STAMINA_SMOOTH_SPEED * delta)
	_right_stamina_display = lerpf(_right_stamina_display, _right_stamina_target, STAMINA_SMOOTH_SPEED * delta)
	player_stamina_bar.value = _left_stamina_display
	right_stamina_bar.value = _right_stamina_display
	_update_stamina_style(player_stamina_bar, _left_stamina_display, player_stamina_bar.max_value)
	_update_stamina_style(right_stamina_bar, _right_stamina_display, right_stamina_bar.max_value)
	_update_move_labels(delta)


func setup_fight(left: Boxer, right: Boxer, vs_cpu: bool, left_name: String, right_name_text: String) -> void:
	_vs_cpu = vs_cpu
	_left_boxer = left
	_right_boxer = right
	_left_name = left_name
	_right_name = right_name_text
	player_label.text = left_name.to_upper()
	right_label.text = right_name_text.to_upper()
	controls_hint.text = _controls_text(vs_cpu)

	left.health.health_changed.connect(_on_left_health)
	right.health.health_changed.connect(_on_right_health)
	left.stamina.stamina_changed.connect(_on_left_stamina)
	right.stamina.stamina_changed.connect(_on_right_stamina)
	_on_left_health(left.health.current_health, left.health.max_health)
	_on_right_health(right.health.current_health, right.health.max_health)
	_on_left_stamina(left.stamina.current_stamina, left.stamina.max_stamina)
	_on_right_stamina(right.stamina.current_stamina, right.stamina.max_stamina)
	_left_stamina_display = left.stamina.current_stamina
	_right_stamina_display = right.stamina.current_stamina


func _controls_text(vs_cpu: bool) -> String:
	if vs_cpu:
		return "P1: WASD move | J/K/L punch | Space guard | Shift dodge"
	return "P1: WASD | J/K/L | Space | Shift  —  P2: Arrows | U/I/O | M guard | Right Shift dodge"


func update_timer(seconds_left: float) -> void:
	var total_secs := maxi(0, int(seconds_left))
	var mins := total_secs / 60
	var secs := total_secs % 60
	timer_label.text = "%d:%02d" % [mins, secs]


func update_score(left: int, right: int) -> void:
	score_label.visible = DEBUG_SHOW_SCORE
	if DEBUG_SHOW_SCORE:
		score_label.text = "%s %d  —  %d %s" % [_left_name, left, right, _right_name]


func update_debug_distance(distance: float, show_debug: bool) -> void:
	debug_distance_label.visible = show_debug
	if show_debug:
		debug_distance_label.text = "Distance: %d px" % int(distance)


func show_feedback(text: String, duration: float = 1.2) -> void:
	feedback_label.text = text
	feedback_timer = duration


func show_move(move_name: String, is_left: bool) -> void:
	var label := _left_move_label if is_left else _right_move_label
	label.text = move_name.to_upper()
	label.visible = true
	if is_left:
		_left_move_timer = MOVE_LABEL_TIME
	else:
		_right_move_timer = MOVE_LABEL_TIME


func show_result(text: String) -> void:
	result_label.text = text
	result_panel.visible = true


func _on_left_health(current: int, maximum: int) -> void:
	player_health_bar.max_value = maximum
	player_health_bar.value = current


func _on_right_health(current: int, maximum: int) -> void:
	right_health_bar.max_value = maximum
	right_health_bar.value = current


func _on_left_stamina(current: float, maximum: float) -> void:
	player_stamina_bar.max_value = maximum
	_left_stamina_target = current


func _on_right_stamina(current: float, maximum: float) -> void:
	right_stamina_bar.max_value = maximum
	_right_stamina_target = current


func _style_bar(bar: ProgressBar, fill_color: Color) -> void:
	var bg := StyleBoxFlat.new()
	bg.bg_color = Color(0.06, 0.07, 0.1, 0.92)
	bg.corner_radius_top_left = 4
	bg.corner_radius_top_right = 4
	bg.corner_radius_bottom_left = 4
	bg.corner_radius_bottom_right = 4
	bar.add_theme_stylebox_override("background", bg)
	var fill := StyleBoxFlat.new()
	fill.bg_color = fill_color
	fill.corner_radius_top_left = 4
	fill.corner_radius_top_right = 4
	fill.corner_radius_bottom_left = 4
	fill.corner_radius_bottom_right = 4
	bar.add_theme_stylebox_override("fill", fill)


func _update_stamina_style(bar: ProgressBar, current: float, maximum: float) -> void:
	var ratio := current / maxf(maximum, 1.0)
	var color := STAMINA_LOW_COLOR if ratio <= 0.25 else STAMINA_NORMAL_COLOR
	_style_bar(bar, color)


func _make_move_label(color: Color) -> Label:
	var label := Label.new()
	label.visible = false
	label.custom_minimum_size = Vector2(170, 28)
	label.add_theme_color_override("font_color", color)
	label.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.85))
	label.add_theme_constant_override("shadow_offset_x", 2)
	label.add_theme_constant_override("shadow_offset_y", 2)
	label.add_theme_font_size_override("font_size", 22)
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	return label


func _update_move_labels(delta: float) -> void:
	_left_move_timer = maxf(0.0, _left_move_timer - delta)
	_right_move_timer = maxf(0.0, _right_move_timer - delta)
	_left_move_label.visible = _left_move_timer > 0.0
	_right_move_label.visible = _right_move_timer > 0.0
	if _left_boxer:
		_left_move_label.position = _left_boxer.global_position + Vector2(-85, -205)
	if _right_boxer:
		_right_move_label.position = _right_boxer.global_position + Vector2(-85, -205)


func _on_restart_pressed() -> void:
	restart_requested.emit()


func _on_menu_pressed() -> void:
	menu_requested.emit()

#!/usr/bin/env bash
# Kestrel launcher (Linux). Layers presets/<preset>.ini, kestrel.ini and kestrel.local.ini,
# maps the result to kyty_emulator flags, and optionally records a pasteable test report.
# Behaves identically to kestrel.ps1 on Windows. Needs bash 4.3+, coreutils and awk.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CALLER_DIR=$PWD

# ---------------------------------------------------------------- state
declare -A CFG=()          # effective settings, key = section.key (lower case)
declare -a ARGS=()         # emulator arguments (one argv element each)
declare -a CONFIG_FILES=() # files that contributed, for display
HELP_TEXT=""
BANNER=""
ASSUME_ALL=0               # dry-run without a built emulator: assume every flag exists
EMU=""
EMU_DIR=""
PRESET=""
GUEST_LOG_PATH=""
COMING_SOON=" --upscaler --fsr-sharpness --patch --perf-overlay --frame-time-log "

# ---------------------------------------------------------------- helpers
die() { echo "kestrel: error: $*" >&2; exit 2; }
warn() { echo "warning: $*" >&2; }

usage() {
	cat <<'EOF'
Usage: ./kestrel.sh [options]

  --dry-run             print the effective settings and the emulator command, do not launch
  --report [SECONDS]    run the game for SECONDS (default 300), then print a summary block
                        to paste back to the developers
  --preset NAME         use presets/NAME.ini instead of the preset in kestrel.ini
  --config FILE         use FILE instead of kestrel.ini
  --game DIR            game folder (overrides [game] path)
  --help                this text

Settings live in kestrel.ini (commented). Put personal overrides in kestrel.local.ini
(ignored by git). See docs/QUICKSTART.md for setup.
EOF
}

# trim: strip leading/trailing whitespace of $1 into TRIMMED (avoids a subshell per call).
TRIMMED=""
trim() {
	local s=$1
	s=${s#"${s%%[![:space:]]*}"}
	s=${s%"${s##*[![:space:]]}"}
	TRIMMED=$s
}

# ini_overlay FILE ARRAY_NAME: parse FILE into the associative array; non-empty values override.
ini_overlay() {
	local file=$1
	local -n target=$2
	local line section="" key val
	while IFS= read -r line || [[ -n $line ]]; do
		line=${line%$'\r'}
		line=${line#$'\xef\xbb\xbf'}
		trim "$line"; line=$TRIMMED
		[[ -z $line || $line == \;* || $line == \#* ]] && continue
		if [[ $line == \[*\] ]]; then
			section=${line:1:${#line}-2}
			trim "$section"; section=${TRIMMED,,}
			continue
		fi
		[[ $line == *=* ]] || continue
		trim "${line%%=*}"; key=${TRIMMED,,}
		trim "${line#*=}"; val=$TRIMMED
		# shellcheck disable=SC2034 # nameref: writes through to the caller's array
		if [[ -n $val && -n $key ]]; then target["$section.$key"]=$val; fi
	done <"$file"
}

# is_on KEY: switch value -> 0 (on) / 1 (off); anything else is an error naming the key.
is_on() {
	local v=${CFG[$1]:-}
	v=${v,,}
	case $v in
		on | true | yes | 1) return 0 ;;
		off | false | no | 0 | "") return 1 ;;
		*) die "$1 must be on/off (true/false, yes/no, 1/0), got '${CFG[$1]}'" ;;
	esac
}

# is_nonzero VALUE: true when VALUE is non-empty and not 0.
is_nonzero() { [[ -n $1 && $1 != 0 ]]; }

# abs_path PATH BASE: PATH made absolute against BASE (no symlink resolution).
abs_path() { if [[ $1 == /* ]]; then printf '%s' "$1"; else printf '%s/%s' "$2" "$1"; fi; }

# quote_arg ARG: shell-quote only when needed, for the printed copy-pasteable command.
quote_arg() {
	if [[ -n $1 && $1 != *[!A-Za-z0-9_@%+=:,./-]* ]]; then
		printf '%s' "$1"
	else
		printf "'%s'" "${1//\'/\'\\\'\'}"
	fi
}

format_cmd() {
	local out="" a
	for a in "$@"; do out+="$(quote_arg "$a") "; done
	printf '%s' "${out% }"
}

# ---------------------------------------------------------------- config
load_config() {
	local base_cfg=$1 preset_cli=$2
	declare -A boot=()
	[[ -f $base_cfg ]] || die "config file not found: $base_cfg"
	ini_overlay "$base_cfg" boot
	[[ -f $SCRIPT_DIR/kestrel.local.ini ]] && ini_overlay "$SCRIPT_DIR/kestrel.local.ini" boot
	PRESET=${preset_cli:-${boot[launcher.preset]:-default}}

	local preset_file=$SCRIPT_DIR/presets/$PRESET.ini
	if [[ ! -f $preset_file ]]; then
		local avail="" f
		for f in "$SCRIPT_DIR"/presets/*.ini; do
			[[ -f $f ]] && avail+=" $(basename "$f" .ini)"
		done
		die "unknown preset '$PRESET'; available:${avail:- (none found in presets/)}"
	fi
	ini_overlay "$preset_file" CFG;  CONFIG_FILES+=("$preset_file")
	ini_overlay "$base_cfg" CFG;     CONFIG_FILES+=("$base_cfg")
	if [[ -f $SCRIPT_DIR/kestrel.local.ini ]]; then
		ini_overlay "$SCRIPT_DIR/kestrel.local.ini" CFG
		CONFIG_FILES+=("$SCRIPT_DIR/kestrel.local.ini")
	fi
}

# ---------------------------------------------------------------- emulator
find_emulator() {
	local dry=$1 cand
	local configured=${CFG[launcher.emulator]:-}
	if [[ -n $configured ]]; then
		EMU=$(abs_path "$configured" "$SCRIPT_DIR")
		[[ -f $EMU ]] || die "[launcher] emulator not found: $EMU"
	else
		for cand in _Build/linux/install/kyty_emulator _Build/linux/kyty_emulator; do
			if [[ -f $SCRIPT_DIR/$cand ]]; then EMU=$SCRIPT_DIR/$cand; break; fi
		done
	fi
	if [[ -z $EMU ]]; then
		[[ $dry == 1 ]] || die "kyty_emulator not found; build first, see docs/QUICKSTART.md"
		warn "kyty_emulator not found (build first, see docs/QUICKSTART.md); using placeholder, assuming all flags are supported"
		EMU=kyty_emulator
		EMU_DIR=$SCRIPT_DIR
		ASSUME_ALL=1
		BANNER="(emulator not built)"
		return
	fi
	EMU_DIR=$(cd "$(dirname "$EMU")" && pwd)
	EMU=$EMU_DIR/$(basename "$EMU")
}

detect_features() {
	[[ $ASSUME_ALL == 1 ]] && return
	if command -v timeout >/dev/null 2>&1; then
		HELP_TEXT=$(timeout 15 "$EMU" --help 2>&1 </dev/null || true)
	else
		HELP_TEXT=$("$EMU" --help 2>&1 </dev/null || true)
	fi
	BANNER=${HELP_TEXT%%$'\n'*}
}

# supports FLAG: whole-word match followed by whitespace/end in the help text.
supports() {
	[[ $ASSUME_ALL == 1 ]] && return 0
	[[ $HELP_TEXT =~ (^|[[:space:]])$1([[:space:]]|$) ]]
}

# need KEY FLAG: true if FLAG is supported, else print the skip warning and return 1.
need() {
	supports "$2" && return 0
	if [[ $COMING_SOON == *" $2 "* ]]; then
		warn "$1 needs $2 (coming soon, not in this build); skipped"
	else
		warn "$1 needs $2 (not in this build); skipped"
	fi
	return 1
}

# ---------------------------------------------------------------- argument mapping
# build_args GAME_DIR DRY: fill ARGS from CFG. Never emits a flag the emulator lacks.
build_args() {
	local game=$1 dry=$2 v
	ARGS=()

	if [[ -n $game ]]; then ARGS+=(--game "$game"); else ARGS+=(--game "<game dir>"); fi

	# --redzone exists only in Windows builds: Linux signal delivery already skips the guest red
	# zone, so the setting is simply not needed here.
	# (No warning: the Demon's Souls presets turn it on for Windows.)
	is_on game.redzone || true # still validate the value

	v=${CFG[video.render_scale]:-}
	[[ -n $v ]] && need video.render_scale --render-scale && ARGS+=(--render-scale "$v")
	if is_on video.fullscreen && need video.fullscreen --fullscreen; then ARGS+=(--fullscreen); fi
	v=${CFG[video.width]:-}
	[[ -n $v ]] && need video.width --screen-width && ARGS+=(--screen-width "$v")
	v=${CFG[video.height]:-}
	[[ -n $v ]] && need video.height --screen-height && ARGS+=(--screen-height "$v")
	v=${CFG[video.rt_mode]:-}
	[[ -n $v ]] && need video.rt_mode --rt-mode && ARGS+=(--rt-mode "$v")
	v=${CFG[video.vertex_fetch]:-}
	[[ -n $v ]] && need video.vertex_fetch --vertex-fetch && ARGS+=(--vertex-fetch "$v")
	v=${CFG[video.present_mode]:-}
	[[ -n $v ]] && need video.present_mode --present-mode && ARGS+=(--present-mode "$v")

	map_upscaler

	local frame_cap=${CFG[speed.frame_cap]:-}
	map_game_speed_patch frame_cap
	is_nonzero "$frame_cap" && need speed.frame_cap --frame-cap && ARGS+=(--frame-cap "$frame_cap")

	if is_on debug.overlay; then
		if supports --perf-overlay; then
			ARGS+=(--perf-overlay)
		else
			warn "debug.overlay needs --perf-overlay (coming soon, not in this build); skipped; the window title shows fps instead"
		fi
	fi
	v=${CFG[debug.frame_time_log]:-}
	if [[ -n $v ]] && need debug.frame_time_log --frame-time-log; then
		ARGS+=(--frame-time-log "$(abs_path "$v" "$SCRIPT_DIR")")
	fi
	v=${CFG[debug.fps_log]:-}
	is_nonzero "$v" && need debug.fps_log --fps-log && ARGS+=(--fps-log "$v")

	if is_on debug.guest_log; then
		if [[ -z $GUEST_LOG_PATH ]]; then
			GUEST_LOG_PATH=$SCRIPT_DIR/kestrel-logs/$(date +%Y%m%d-%H%M%S).guest.log
		fi
		[[ $dry == 1 ]] || mkdir -p "$(dirname "$GUEST_LOG_PATH")"
		if need debug.guest_log --printf-direction && need debug.guest_log --printf-output-file; then
			ARGS+=(--printf-direction File --printf-output-file "$GUEST_LOG_PATH")
		fi
	fi

	v=${CFG[debug.extra_args]:-}
	if [[ -n $v ]]; then
		local -a extra
		read -r -a extra <<<"$v"
		ARGS+=("${extra[@]}")
	fi
}

map_upscaler() {
	local u=${CFG[video.upscaler]:-}
	u=${u,,}
	case $u in
		"" | linear)
			supports --upscaler && ARGS+=(--upscaler linear)
			;;
		fsr1)
			if need video.upscaler --upscaler; then
				ARGS+=(--upscaler fsr1)
				local s=${CFG[video.fsr_sharpness]:-}
				[[ -n $s ]] && need video.fsr_sharpness --fsr-sharpness && ARGS+=(--fsr-sharpness "$s")
			fi
			;;
		*) die "video.upscaler must be linear or fsr1, got '${CFG[video.upscaler]}'" ;;
	esac
	return 0
}

# map_game_speed_patch FRAME_CAP_VAR: may add --patch and update the caller's frame cap variable.
map_game_speed_patch() {
	local -n cap=$1
	is_on speed.game_speed_patch || return 0
	local name=${CFG[speed.patch_name]:-}
	if [[ -z $name ]]; then
		warn "speed.game_speed_patch is on but there is no game-speed patch for this preset; skipped"
		return 0
	fi
	if need speed.game_speed_patch --patch; then
		ARGS+=(--patch "$name")
		local pcap=${CFG[speed.patch_frame_cap]:-}
		if ! is_nonzero "$cap" && is_nonzero "$pcap"; then cap=$pcap; fi
	fi
	return 0
}

# ---------------------------------------------------------------- report mode
# frametime_stats FILE WINDOW_US LABEL: stats over intervals between consecutive lines.
frametime_stats() {
	awk -v win="$2" 'NF >= 2 { t[++n] = $2 + 0 }
		END { last = t[n]; for (i = 2; i <= n; i++) if (win == 0 || t[i] >= last - win) print t[i] - t[i-1] }' "$1" |
		sort -rn |
		awk -v label="$3" '
		function mean_top(p,   k, i, s) { k = int(n * p); if (k < n * p) k++; if (k < 1) k = 1
			for (i = 1; i <= k; i++) s += a[i]; return s / k }
		{ a[NR] = $1; sum += $1; if ($1 > 33300) slow++ }
		END {
			n = NR
			if (n < 1 || sum <= 0) { printf "  %s: no frames\n", label; exit }
			printf "  %s: frames=%d avg_fps=%.1f 1%%low_fps=%.1f 10%%low_fps=%.1f max_frame_ms=%.1f over_33.3ms=%.1f%%\n",
				label, n, n / sum * 1e6, 1e6 / mean_top(0.01), 1e6 / mean_top(0.10), a[1] / 1000, slow * 100 / n }'
}

# fps_log_stats GUEST_LOG: per-second fps samples from "fps: N frame: N t=Ns" lines.
fps_log_stats() {
	tr -d '\r' <"$1" | awk '
		/^fps: [0-9]+ frame: [0-9]+ t=[0-9]+s/ { f[++n] = $2 + 0; fr = $4 + 0 }
		function stat(label, from,   i, c, s, mn, mx) {
			c = 0; s = 0; mn = 1e18; mx = -1
			for (i = from; i <= n; i++) { c++; s += f[i]; if (f[i] < mn) mn = f[i]; if (f[i] > mx) mx = f[i] }
			printf "  %s: samples=%d avg_fps=%.1f min_fps=%d max_fps=%d\n", label, c, s / c, mn, mx }
		END {
			if (n == 0) exit 1
			stat("whole run", 1); stat("last 60 samples", n > 60 ? n - 59 : 1)
			printf "  last frame number: %d\n", fr }'
}

print_frame_stats() {
	local dir=$1 ft=$1/frametimes.txt gl=$1/guest.log
	if [[ -f $ft && $(wc -l <"$ft") -ge 3 ]]; then
		echo "Frame stats (frame-time log, $(wc -l <"$ft") lines):"
		frametime_stats "$ft" 0 "whole run"
		frametime_stats "$ft" 60000000 "last 60 s"
	elif [[ -f $gl ]] && grep -qE '^fps: [0-9]+ frame: [0-9]+ t=[0-9]+s' "$gl"; then
		echo "Frame stats (guest log fps counters):"
		fps_log_stats "$gl"
	else
		echo "Frame stats: no frame data (the game did not reach rendering?)"
	fi
}

sanitize_args() {
	# Replace values that are paths with their last component (privacy).
	local -a out=()
	local i=0 n=$#
	local -a in=("$@")
	while ((i < n)); do
		case ${in[i]} in
			--game | --printf-output-file | --frame-time-log | --game-patch)
				out+=("${in[i]}")
				if ((i + 1 < n)); then out+=("$(basename "${in[i+1]}")"); fi
				i=$((i + 2))
				;;
			*) out+=("${in[i]}"); i=$((i + 1)) ;;
		esac
	done
	format_cmd "${out[@]}"
}

system_info() {
	local os cpu ram gpu
	# shellcheck disable=SC1091
	os=$( (. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-Linux}") || echo Linux)
	os="$os, kernel $(uname -r)"
	cpu=$(awk -F: '/model name/ { sub(/^ +/, "", $2); print $2; exit }' /proc/cpuinfo 2>/dev/null || true)
	ram=$(awk '/MemTotal/ { printf "%.1f GiB", $2 / 1048576 }' /proc/meminfo 2>/dev/null || true)
	if command -v nvidia-smi >/dev/null 2>&1; then
		gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)
	fi
	if [[ -z ${gpu:-} ]] && command -v lspci >/dev/null 2>&1; then
		gpu=$(lspci 2>/dev/null | grep -iE 'vga|3d' || true)
	fi
	echo "OS: $os"
	echo "CPU: ${cpu:-unknown}"
	echo "RAM: ${ram:-unknown}"
	echo "GPU: ${gpu:-unknown}"
}

kestrel_commit() {
	local c
	c=$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || true)
	if [[ -z $c ]]; then echo unknown; return; fi
	if [[ -n $(git -C "$SCRIPT_DIR" status --porcelain -uno 2>/dev/null || true) ]]; then c+=" (modified)"; fi
	echo "$c"
}

REPORT_PID=""
stop_emulator() {
	[[ -n $REPORT_PID ]] && kill -0 "$REPORT_PID" 2>/dev/null || return 0
	kill "$REPORT_PID" 2>/dev/null || true
	local i
	for ((i = 0; i < 10; i++)); do
		kill -0 "$REPORT_PID" 2>/dev/null || break
		sleep 1
	done
	kill -0 "$REPORT_PID" 2>/dev/null && kill -9 "$REPORT_PID" 2>/dev/null || true
	wait "$REPORT_PID" 2>/dev/null || true
}

# shellcheck disable=SC2329 # invoked via trap
on_interrupt() {
	echo
	echo "kestrel: interrupted; stopping the emulator" >&2
	trap - INT TERM
	stop_emulator
	exit 130
}

run_report() {
	local seconds=$1 game=$2 stamp dir
	stamp=$(date +%Y%m%d-%H%M%S)
	dir=$SCRIPT_DIR/kestrel-logs/report-$stamp
	mkdir -p "$dir"

	# Force the report settings on top of the config.
	CFG[debug.guest_log]=on
	GUEST_LOG_PATH=$dir/guest.log
	CFG[debug.fps_log]=1
	if supports --frame-time-log; then CFG[debug.frame_time_log]=$dir/frametimes.txt; else CFG[debug.frame_time_log]=""; fi
	build_args "$game" 0
	print_header
	echo "command: $(format_cmd "$EMU" "${ARGS[@]}")"

	echo "Recording $seconds seconds; play normally. The game will be closed automatically."
	local start now elapsed=0 ended="" code=0 next_progress=30
	start=$(date +%s)
	(cd "$EMU_DIR" && exec "$EMU" "${ARGS[@]}") >"$dir/emulator.log" 2>&1 </dev/null &
	REPORT_PID=$!
	trap on_interrupt INT TERM

	while ((elapsed < seconds)); do
		if ! kill -0 "$REPORT_PID" 2>/dev/null; then
			wait "$REPORT_PID" || code=$?
			ended="exited early (code $code)"
			break
		fi
		sleep 1
		now=$(date +%s); elapsed=$((now - start))
		if ((elapsed >= next_progress && elapsed < seconds)); then
			echo "  ... ${elapsed}s of ${seconds}s"
			next_progress=$((next_progress + 30))
		fi
	done
	if [[ -z $ended ]]; then
		if kill -0 "$REPORT_PID" 2>/dev/null; then
			stop_emulator
			ended="timer"
		else
			wait "$REPORT_PID" || code=$?
			ended="exited early (code $code)"
		fi
	fi
	trap - INT TERM
	REPORT_PID=""
	now=$(date +%s); elapsed=$((now - start))

	local early=0
	[[ $ended == exited* ]] && early=1
	{
		echo "----- Kestrel report (paste everything below) -----"
		echo "kestrel commit: $(kestrel_commit)"
		echo "emulator: $BANNER"
		system_info
		echo "preset: $PRESET"
		echo "game folder: $(basename "${game:-<none>}")"
		echo "arguments: $(sanitize_args "${ARGS[@]}")"
		echo "duration: requested ${seconds}s, actual ${elapsed}s"
		echo "ended: $ended"
		print_frame_stats "$dir"
		if ((early)); then
			echo "Last 15 lines of emulator.log:"
			tail -n 15 "$dir/emulator.log" | sed 's/^/  | /'
		fi
		echo "report folder: $dir"
		echo "----- end of report -----"
	}
}

# ---------------------------------------------------------------- main
print_header() {
	echo "preset: $PRESET"
	local f
	for f in "${CONFIG_FILES[@]}"; do echo "config: $f"; done
	echo "emulator: $BANNER"
	echo "saves: $EMU_DIR/_SaveData"
}

main() {
	local dry=0 report=0 seconds=300 preset_cli="" config="" game_cli=""
	while (($# > 0)); do
		case $1 in
			--dry-run) dry=1 ;;
			--report)
				report=1
				if (($# > 1)) && [[ $2 =~ ^[0-9]+$ ]]; then seconds=$2; shift; fi
				((seconds > 0)) || die "--report SECONDS must be greater than 0"
				;;
			--preset) (($# > 1)) || die "--preset needs a name"; preset_cli=$2; shift ;;
			--config) (($# > 1)) || die "--config needs a file"; config=$2; shift ;;
			--game) (($# > 1)) || die "--game needs a folder"; game_cli=$2; shift ;;
			--help | -h) usage; exit 0 ;;
			*) usage >&2; die "unknown option: $1" ;;
		esac
		shift
	done

	local base_cfg=$SCRIPT_DIR/kestrel.ini
	[[ -n $config ]] && base_cfg=$(abs_path "$config" "$CALLER_DIR")
	load_config "$base_cfg" "$preset_cli"
	find_emulator "$dry"
	detect_features

	# Game folder: CLI path is relative to the caller's cwd, the ini path to the script dir.
	local game=""
	if [[ -n $game_cli ]]; then
		game=$(abs_path "$game_cli" "$CALLER_DIR")
	elif [[ -n ${CFG[game.path]:-} ]]; then
		game=$(abs_path "${CFG[game.path]}" "$SCRIPT_DIR")
	fi
	if [[ -z $game ]]; then
		[[ $dry == 1 ]] || die "no game folder; set [game] path in kestrel.ini / kestrel.local.ini or pass --game DIR"
		warn "no game folder set; using placeholder <game dir>"
	elif [[ ! -d $game ]]; then
		if [[ $dry == 1 ]]; then warn "game folder does not exist: $game"; else die "game folder does not exist: $game"; fi
	fi

	if ((report)) && ((! dry)); then
		run_report "$seconds" "$game"
		exit 0
	fi

	print_header
	build_args "$game" "$dry"
	echo "command: $(format_cmd "$EMU" "${ARGS[@]}")"
	if ((dry)); then exit 0; fi

	local rc=0
	(cd "$EMU_DIR" && exec "$EMU" "${ARGS[@]}") || rc=$?
	exit "$rc"
}

main "$@"

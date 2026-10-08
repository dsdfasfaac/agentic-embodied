#!/usr/bin/env bash
set -euo pipefail

# Manage dodo's two ARX remote_slave ROS2 controllers as one process group.
# Starting this service can enable motors; use only with an on-site observer.
# stop disables the motors; it does not home. For ordinary completed rollouts,
# use finish_arx_real_episode.py to verify homing before invoking this action.
ACTION="${1:-status}"
ROOT="${ARX_DODO_ROOT:-/home/dodo/chenfu/Agentic-Embodied}"
ARX_WS="${ARX_DODO_WS:-/home/dodo/chenfu/ARX_X5/ROS2/X5_ws}"
STATE_DIR="${ARX_DODO_STATE_DIR:-$ROOT/runs/arx_live_readonly_20261004/controller}"
PID_FILE="$STATE_DIR/ros2_joint_control.pid"
LOG_FILE="$STATE_DIR/ros2_joint_control.log"

running() {
  [[ -s "$PID_FILE" ]] || return 1
  local pid
  pid="$(<"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

case "$ACTION" in
  status)
    if running; then
      echo "running pid=$(<"$PID_FILE") log=$LOG_FILE"
    else
      echo "not running"
      exit 1
    fi
    ;;
  start)
    if running; then
      "$0" status
      exit 0
    fi
    for iface in can1 can3; do
      ip link show "$iface" | grep -q 'state UP' || {
        echo "$iface is not UP" >&2; exit 2;
      }
    done
    if pgrep -x X5Controller >/dev/null; then
      echo "another X5Controller is already running" >&2; exit 2
    fi
    mkdir -p "$STATE_DIR"
    rm -f "$PID_FILE"
    setsid bash -c '
      set -eo pipefail
      echo $$ > "$1"
      source /opt/ros/jazzy/setup.bash
      source "$2/install/setup.bash"
      set -u
      exec ros2 launch arx_x5_controller v2_joint_control.launch.py
    ' arx-dodo-controller "$PID_FILE" "$ARX_WS" >>"$LOG_FILE" 2>&1 </dev/null &
    for _ in {1..100}; do
      if running && pgrep -x X5Controller >/dev/null; then
        "$0" status
        exit 0
      fi
      if [[ -s "$PID_FILE" ]] && ! running; then
        tail -n 30 "$LOG_FILE" >&2
        exit 1
      fi
      sleep 0.1
    done
    echo "ARX ROS2 controller did not start; log=$LOG_FILE" >&2
    exit 1
    ;;
  stop)
    if running; then
      pid="$(<"$PID_FILE")"
      kill -TERM -- "-$pid"
      for _ in {1..50}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
      done
      if kill -0 "$pid" 2>/dev/null; then
        echo "controller process group still running after TERM: $pid" >&2
        exit 1
      fi
      echo "stopped pid=$pid"
    else
      echo "not running"
    fi
    rm -f "$PID_FILE"
    ;;
  tail)
    tail -n 100 -f "$LOG_FILE"
    ;;
  *)
    echo "usage: $0 {start|status|stop|tail}" >&2
    exit 2
    ;;
esac

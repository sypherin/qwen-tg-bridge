#!/usr/bin/env bash
# systemd ExecStart for qwen-tg-bridge.service (Type=oneshot).
# Spawns the qwentg tmux session if it isn't already running; idempotent, so
# a `systemctl --user restart qwen-tg-bridge` (or boot) never double-spawns
# and never kills a live bridge mid-run.
set -euo pipefail
SESSION=qwentg
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session $SESSION already running — leaving it alone"
    exit 0
fi
tmux new-session -d -s "$SESSION" -x 220 -y 50 "$HOME/qwen-tg-bridge/run-in-tmux.sh"
sleep 2
tmux has-session -t "$SESSION"
echo "tmux session $SESSION started — attach with: tmux attach -t $SESSION"

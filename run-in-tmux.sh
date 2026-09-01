#!/usr/bin/env bash
# Pane runner for the qwentg tmux session.
# The loop preserves the old systemd Restart=always semantics INSIDE tmux:
# if bridge.py crashes (e.g. a telegram httpx flake), it restarts after 5s and
# the crash + traceback stay visible in the scrollback instead of vanishing
# into journald. The pane never exits, so the session never silently dies.
cd "$(dirname "$0")"
while true; do
    echo "=== $(date '+%F %T') bridge starting ==="
    ./start.sh
    echo "=== $(date '+%F %T') bridge exited rc=$? — restarting in 5s ==="
    sleep 5
done

#!/usr/bin/env bash
# Launch the qwen<->Telegram bridge. Captures your chat_id from the bot's
# recent messages on first run, writes it to the env file, then starts polling.
set -euo pipefail
ENV=~/.env.qwen-tg-bridge
set -a; source "$ENV"; set +a

if [ -z "${QWEN_TG_ALLOWED_CHATS:-}" ]; then
  echo "No allowlist yet — reading your chat_id from the bot's recent messages (send your bot a DM first)..."
  CHAT=$(python3 -c "
import urllib.request,json,os
t=os.environ['TG_QWEN_BOT_TOKEN']
u=json.load(urllib.request.urlopen('https://api.telegram.org/bot'+t+'/getUpdates?timeout=1',timeout=8))
ids=[ (m.get('message') or m.get('edited_message') or {}).get('chat',{}).get('id') for m in u.get('result',[]) ]
ids=[str(i) for i in ids if i]
print(ids[-1] if ids else '')
")
  if [ -z "$CHAT" ]; then echo "  -> No message found. Send your bot a DM first, then re-run."; exit 1; fi
  echo "QWEN_TG_ALLOWED_CHATS=$CHAT" >> "$ENV"
  export QWEN_TG_ALLOWED_CHATS="$CHAT"
  echo "  -> allowlisted chat_id $CHAT"
fi

# 2026-09-01: synced with the systemd unit values — the bridge brain is the
# :8022 llama-server (Qwen3.8-27B), not the old :8001/qwen3.6. Also export PATH
# so the `qwen` CLI (~/.npm-global/bin) is reachable when spawned from tmux,
# whose server env does not inherit the login shell's PATH.
export PATH="$HOME/.npm-global/bin:$PATH"
export OPENAI_BASE_URL=http://127.0.0.1:8022/v1
export OPENAI_API_KEY=sk-local
export OPENAI_MODEL=qwen3.8-27b
exec python3 ~/qwen-tg-bridge/bridge.py

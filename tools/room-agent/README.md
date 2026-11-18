# room-agent

Voice conversation with Claude Code through a Xiaomi camera's microphone and the 小爱 speaker
in the same room. Experimental; built on the Miloco backend running from source.

```
camera mic ──► silero VAD ──► mlx-whisper ──► Claude Code session ──► tools/room-agent/say ──► 小爱 speaker
camera video ─► 30 s frame buffer ─► frame attached to each message
camera events (cloud MQTT) ─► optional "someone entered" message
```

## How it works

- **Continuous local buffer.** The agent keeps the camera stream open and holds the last 30 s of
  JPEG frames and 16 kHz audio in memory. Nothing is written to disk except the frames attached
  to a message (`$MILOCO_HOME/room-agent/captures/`).
- **Wake word.** Speech is transcribed locally with Whisper. Saying "Claude" / "克劳德" opens a
  conversation; while it is open (3 min after the last exchange, or after Claude speaks),
  everything said is forwarded and Claude decides whether it was addressed to it.
- **Talking back.** Claude replies with `tools/room-agent/say --room <room> "<text>"`, which
  speaks through the 小爱 speaker assigned to that room in Mi Home.
- **Barge-in.** The mic stays open while the speaker talks; transcripts that mostly match what
  Claude is saying are dropped as echo, anything else is forwarded as a possible interruption.
- **Into the session.** Messages are posted to the Claude Code session's cross-session inbox
  (`$CLAUDE_CODE_MESSAGING_SOCKET`), so the agent must be started from inside that session.
  The message line format (`{"type":"user","message":{...}}` after the auth line) is not part
  of the documented interface and may change.

## Setup

1. Run Miloco from this checkout (no installer, no OpenClaw/Hermes); state lives in
   `.miloco-home/` (git-ignored):
   ```bash
   bin/miloco-backend start
   bin/miloco-cli account bind
   ```
2. Install Whisper into the backend venv (Apple silicon; not part of the lockfile, so a later
   `uv sync` removes it):
   ```bash
   uv pip install --python backend/.venv/bin/python mlx-whisper
   ```
   The first run downloads `mlx-community/whisper-large-v3-turbo` (~1.6 GB).
3. Make sure the camera is switched on in Mi Home, then from inside a Claude Code session:
   ```bash
   bin/room-agent start <camera did> [<camera did> ...]
   bin/room-agent logs
   bin/room-agent stop
   ```

Options (`tools/room-agent/room_agent.py --help`): `--language` (Whisper language, default
`zh`; `auto` to detect), `--reply-language` (default `Chinese`), `--greet-on-entry`.
`MILOCO_CLOUD_SERVER` selects the Xiaomi region (default `cn`).

## Limitations

- Only rooms with a working camera can hear you; speakers in other rooms can still talk.
- Expect ~5–10 s per reply: VAD end-of-speech (0.5 s), Whisper (~1 s on M4), the Claude turn,
  and the cloud speaker command (~1.5 s). The speaker always finishes its current sentence.
- Anyone in the room can talk to Claude once a conversation is open; confirm risky actions.
- Battery cameras and doorbells (e.g. `midr.*`) did not stream with the bundled camera library.

## Related backend change

`backend/miot/src/miot/client.py` routes actions for `wifispeaker` devices to the cloud: via the
central hub gateway, `play-text` returned code 0 but the speaker stayed silent.

## macOS note

`bin/*` put the source trees on `PYTHONPATH` explicitly: macOS can mark files under `.venv` as
hidden, and Python 3.12+ skips hidden `.pth` files, which silently breaks the editable installs.

"""Room agent: continuous camera A/V buffer + presence + speech, feeding a Claude Code session.

What it does, per camera:
  * keeps the camera stream open and holds the last BUFFER_S seconds of video frames (JPEG)
    and audio (16 kHz PCM) in memory;
  * listens to the camera's own events on Xiaomi's cloud MQTT (someone-appeared / object-motion);
  * when someone enters a room that has been empty for EMPTY_GAP_S, saves the frames around
    that moment and posts an "entered" message to the Claude Code session;
  * runs a voice-activity detector (silero) on the camera microphone, transcribes each
    utterance locally (mlx-whisper) and, while a conversation is active (or the wake word is
    heard), posts what was said to the session;
  * ignores the microphone while a room speaker is talking (see ./say) so Claude doesn't
    hear itself.

Messages go to the session's cross-session inbox ($CLAUDE_CODE_MESSAGING_SOCKET), so the
agent must be started from inside that Claude Code session (bin/room-agent does that).

Run with the backend venv; see bin/room-agent.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import os
import re
import socket
import sqlite3
import ssl
import threading
import time
import urllib.request
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HOME = Path(os.environ.get("MILOCO_HOME", REPO / ".miloco-home"))
STATE = HOME / "room-agent"
CAPTURES = STATE / "captures"
SPEAKING_FILE = STATE / "speaking_until"  # written by ./say
ROOMS_FILE = STATE / "rooms.json"         # room -> speaker did, read by ./say
MODELS = REPO / "backend/miloco/src/miloco/perception/models"

CLOUD_SERVER = os.environ.get("MILOCO_CLOUD_SERVER", "cn")
# MIoT-spec ai-detection events (siid 8) as published by Xiaomi cameras such as the 4C.
EVENTS = {"8/4": "object-motion", "8/5": "someone-appeared"}

BUFFER_S = 30.0          # rolling A/V buffer
EMPTY_GAP_S = 300.0      # room counts as newly entered after this long without a person
CONVERSATION_S = 180.0   # conversation stays open this long after the last exchange
ECHO_TAIL_S = 0.8        # keep ignoring the mic a bit after the speaker stops
# Whisper spells the wake word many ways, in both languages.
WAKE_WORDS = re.compile(
    r"\b(claude|claud|clod|cloud|clode|klaud|claro|clau|clawed)\b|克劳德|克勞德|克劳迪|克罗德|可劳德|克洛德|克劳", re.I)
# Default to Chinese-only decoding: in auto mode Whisper turned short Chinese commands into
# English-ish noise ("关掉" -> "Quanta"). The prompt is simplified Chinese with the wake word
# and generic household phrases, which also keeps output in simplified characters.
WHISPER_LANGUAGE: str | None = "zh"   # --language; "auto" -> None (detect)
WHISPER_PROMPT = "克劳德，帮我把灯关掉。打开客厅的灯，调暗一点。明天天气怎么样？好的，谢谢。"
REPLY_LANGUAGE = "Chinese"           # --reply-language; what Claude is asked to answer in
GREET_ON_ENTRY = False               # --greet-on-entry
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
HALLUCINATIONS = re.compile(
    r"^(thank you\.?|thanks for watching.*|谢谢观看.*|字幕.*|请不吝点赞.*|you)$", re.I)

log = logging.getLogger("room-agent")


# ---------------------------------------------------------------- Miloco / Xiaomi helpers

def miloco_get(path: str):
    cfg = json.loads((HOME / "config.json").read_text())
    req = urllib.request.Request(
        f"http://127.0.0.1:1810{path}",
        headers={"Authorization": f"Bearer {cfg['server']['token']}"})
    return json.load(urllib.request.urlopen(req, timeout=15))["data"]


def access_token() -> str:
    row = sqlite3.connect(HOME / "miloco.db").execute(
        "select value from kv where key='MIOT_TOKEN_INFO_KEY'").fetchone()
    return json.loads(row[0])["access_token"]


def devices() -> list[dict]:
    data = miloco_get("/api/miot/device_list")
    items = data if isinstance(data, list) else (data.get("devices") or list(data.values()))
    return [d for d in items if isinstance(d, dict)]


# ---------------------------------------------------------------- Claude session inbox

def post_to_session(text: str) -> None:
    path = os.environ.get("CLAUDE_CODE_MESSAGING_SOCKET")
    if not path:
        log.warning("no CLAUDE_CODE_MESSAGING_SOCKET; would have posted: %s", text)
        return
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(path)
        lines = [{"type": "auth", "token": os.environ.get("CLAUDE_CODE_MESSAGING_TOKEN", "")},
                 {"type": "user", "message": {"role": "user", "content": text}}]
        s.sendall("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines).encode())
        s.close()
        log.info("posted to session: %s", text[:160])
    except OSError as e:
        log.error("posting to session failed: %s", e)


# ---------------------------------------------------------------- speech: VAD + whisper

class SileroVAD:
    """Streaming silero VAD over 16 kHz int16 PCM (512-sample chunks)."""

    CHUNK, CTX, SR = 512, 64, 16000

    def __init__(self, start=0.5, end=0.35, min_speech_s=0.4, max_silence_s=0.5,
                 max_utt_s=20.0, preroll_s=0.4):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(MODELS / "silero_vad.onnx"), sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.start_th, self.end_th = start, end
        self.min_speech = int(min_speech_s * self.SR)
        self.max_silence = int(max_silence_s * self.SR)
        self.max_utt = int(max_utt_s * self.SR)
        self.preroll = collections.deque(maxlen=int(preroll_s * self.SR / self.CHUNK) + 1)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.ctx = np.zeros((1, self.CTX), dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.int16)
        self.utt: list[np.ndarray] | None = None
        self.silence = 0
        self.preroll.clear()

    def feed(self, pcm: np.ndarray) -> list[np.ndarray]:
        """Feed int16 PCM; returns finished utterances (int16 arrays)."""
        out = []
        self.pending = np.concatenate([self.pending, pcm])
        while self.pending.size >= self.CHUNK:
            chunk, self.pending = self.pending[:self.CHUNK], self.pending[self.CHUNK:]
            x = chunk.astype(np.float32).reshape(1, -1) / 32768.0
            p, self.state = self.sess.run(
                ["output", "stateN"],
                {"input": np.concatenate([self.ctx, x], axis=1), "state": self.state,
                 "sr": np.array(self.SR, dtype=np.int64)})
            self.ctx = x[:, -self.CTX:]
            p = float(p[0, 0])
            if self.utt is None:
                self.preroll.append(chunk)
                if p >= self.start_th:
                    self.utt, self.silence = list(self.preroll), 0
            else:
                self.utt.append(chunk)
                self.silence = self.silence + self.CHUNK if p < self.end_th else 0
                n = sum(c.size for c in self.utt)
                if self.silence >= self.max_silence or n >= self.max_utt:
                    if n - self.silence >= self.min_speech:
                        out.append(np.concatenate(self.utt))
                    self.utt, self.silence = None, 0
                    self.preroll.clear()
        return out


class Transcriber:
    def __init__(self):
        import mlx_whisper  # noqa: F401  (import early so failures show at startup)
        self.pool = ThreadPoolExecutor(max_workers=1)  # mlx: keep it on one thread
        self.pool.submit(self._run, np.zeros(1600, dtype=np.int16)).result()  # warm up

    def _run(self, pcm: np.ndarray) -> str:
        import mlx_whisper
        r = mlx_whisper.transcribe(pcm.astype(np.float32) / 32768.0,
                                   path_or_hf_repo=WHISPER_REPO, temperature=0.0,
                                   condition_on_previous_text=False,
                                   language=WHISPER_LANGUAGE,  # None = auto-detect
                                   initial_prompt=WHISPER_PROMPT)
        segs = r.get("segments", [])
        # compression_ratio > 2.4 is Whisper's own marker for looping output ("我来说" ×50)
        kept = [s for s in segs
                if s.get("no_speech_prob", 0) < 0.6 and s.get("avg_logprob", 0) > -1.0
                and s.get("compression_ratio", 0) <= 2.4]
        if len(kept) < len(segs):
            log.info("dropped low-confidence: %r", [
                (s["text"], round(s.get("no_speech_prob", 0), 2), round(s.get("avg_logprob", 0), 2))
                for s in segs if s not in kept])
        text = "".join(s["text"] for s in kept).strip()
        # collapse any short phrase repeated 3+ times in a row to a single occurrence
        text = re.sub(r"(.{1,12}?)(?:[,，、 ]*\1){2,}", r"\1", text)
        return "" if HALLUCINATIONS.match(text) else text

    async def transcribe(self, pcm: np.ndarray) -> str:
        return await asyncio.get_running_loop().run_in_executor(self.pool, self._run, pcm)


# ---------------------------------------------------------------- per-camera room

class Room:
    def __init__(self, cam: dict, speaker: dict | None, transcriber: Transcriber):
        self.cam, self.speaker, self.transcriber = cam, speaker, transcriber
        self.name = cam.get("room_name") or cam["name"]
        self.frames: collections.deque = collections.deque()  # (ts, jpeg)
        self.audio: collections.deque = collections.deque()   # (ts, int16 pcm)
        self.vad = SileroVAD()
        self.last_person = 0.0
        self.conversation_until = 0.0
        self.entry_task: asyncio.Task | None = None
        self.lock = threading.Lock()

    # -- buffers
    def _trim(self, now):
        while self.frames and now - self.frames[0][0] > BUFFER_S:
            self.frames.popleft()
        while self.audio and now - self.audio[0][0] > BUFFER_S:
            self.audio.popleft()

    async def on_jpg(self, _did, jpg, _ts, _ch):
        now = time.time()
        self.frames.append((now, jpg))
        self._trim(now)

    async def on_pcm(self, _did, data, _ts, _ch):
        now = time.time()
        pcm = np.frombuffer(data, dtype=np.int16)
        self.audio.append((now, pcm))
        self._trim(now)
        su = speaking_until()
        if su > now - ECHO_TAIL_S:
            # Claude just spoke here → the conversation is open (so a plain reply works).
            self.conversation_until = max(self.conversation_until, su + CONVERSATION_S)
        # Keep listening while the speaker talks (barge-in); echo is filtered per utterance.
        for utt in self.vad.feed(pcm):
            asyncio.create_task(self.on_utterance(utt))

    # -- helpers
    def save_frames(self, t0: float, t1: float, tag: str, max_n: int = 4) -> list[str]:
        sel = [(t, j) for t, j in list(self.frames) if t0 <= t <= t1]
        if len(sel) > max_n:
            idx = np.linspace(0, len(sel) - 1, max_n).round().astype(int)
            sel = [sel[i] for i in idx]
        d = CAPTURES / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for t, j in sel:
            p = d / f"{time.strftime('%H%M%S', time.localtime(t))}_{int(t * 1000) % 1000:03d}.jpg"
            p.write_bytes(j)
            paths.append(str(p))
        return paths

    def say_hint(self) -> str:
        if not self.speaker:
            return "(no speaker in this room)"
        return f'reply by voice: tools/room-agent/say --room {self.name} "<text>"'

    # -- presence
    def on_event(self, kind: str):
        now = time.time()
        if kind != "someone-appeared":
            return
        gap = now - self.last_person
        self.last_person = now
        if (GREET_ON_ENTRY and gap >= EMPTY_GAP_S
                and (self.entry_task is None or self.entry_task.done())):
            self.entry_task = asyncio.create_task(self.on_entry(now, gap))

    async def on_entry(self, t_event: float, gap: float):
        await asyncio.sleep(2.0)  # let a couple of post-event frames arrive
        # someone-appeared lands ~6 s after the camera saw the motion, so look back 8 s
        paths = self.save_frames(t_event - 8.0, time.time(), "entry")
        self.conversation_until = time.time() + CONVERSATION_S
        empty = "first sighting since the agent started" if gap > 1e9 else f"empty for {int(gap // 60)} min"
        post_to_session(
            f"[room-agent] {time.strftime('%H:%M:%S', time.localtime(t_event))} someone entered "
            f"{self.name} ({empty}; camera {self.cam['name']}). Frames: {' '.join(paths) or '(none)'}\n"
            f"Look at the frames, then greet them and start a conversation — {self.say_hint()}")

    # -- speech
    async def on_utterance(self, pcm: np.ndarray):
        t_end = time.time()
        dur = pcm.size / 16000
        text = await self.transcriber.transcribe(pcm)
        log.info("[%s] heard (%.1fs): %r", self.name, dur, text)
        if not text:
            return
        overlap = speech_overlap(t_end - dur, t_end)
        if overlap is not None and is_echo(text, overlap):
            log.info("[%s] dropped as echo of the speaker", self.name)
            return
        in_conv = t_end < self.conversation_until
        woke = bool(WAKE_WORDS.search(text))
        if not in_conv and not woke:
            return
        self.last_person = t_end
        self.conversation_until = t_end + CONVERSATION_S
        frame = self.save_frames(t_end - 1.5, t_end, "speech", max_n=1)
        how = "wake word" if woke and not in_conv else "conversation open"
        if overlap is not None:
            how += "; said while you were speaking — they may be interrupting you"
        post_to_session(
            f"[room-agent] {time.strftime('%H:%M:%S', time.localtime(t_end))} heard in {self.name} "
            f"({how}): «{text}»  (frame: {frame[0] if frame else 'none'})\n"
            f"Decide whether this is addressed to you. If it is, answer in {REPLY_LANGUAGE} — "
            f"{self.say_hint()}. If not (people talking to each other, TV), stay silent.")


def speech_overlap(t0: float, t1: float) -> str | None:
    """Text Claude was speaking if [t0, t1] overlaps that speech (plus echo tail), else None."""
    try:
        said = json.loads((STATE / "last_said.json").read_text())
    except (OSError, ValueError):
        return None
    if t0 <= said["until"] + ECHO_TAIL_S and t1 >= said["start"]:
        return said["text"]
    return None


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


def is_echo(heard: str, said: str) -> bool:
    """True if a transcript is (mostly) the speaker's own words picked up by the mic."""
    h, s = _norm(heard), _norm(said)
    if not h:
        return True
    from difflib import SequenceMatcher
    matched = sum(b.size for b in SequenceMatcher(None, h, s, autojunk=False).get_matching_blocks())
    return matched / len(h) >= 0.6


def speaking_until() -> float:
    try:
        return float(SPEAKING_FILE.read_text())
    except (OSError, ValueError):
        return 0.0


# ---------------------------------------------------------------- main

async def main(camera_dids: list[str]):
    from paho.mqtt.client import Client, MQTTv5
    from paho.mqtt.enums import CallbackAPIVersion

    from miot.camera import MIoTCamera
    from miot.const import (
        MIHOME_MQTT_BROKER_HOST_SUFFIX,
        MIHOME_MQTT_PORT,
        OAUTH2_CLIENT_ID,
    )
    from miot.types import MIoTCameraInfo, MIoTCameraStatus, MIoTCameraVideoQuality

    STATE.mkdir(parents=True, exist_ok=True)
    devs = devices()
    by_did = {d["did"]: d for d in devs}
    speakers = {}
    for d in devs:
        if ".wifispeaker." in d.get("model", "") and d.get("room_name"):
            speakers.setdefault(d["room_name"], d)
    ROOMS_FILE.write_text(json.dumps({r: d["did"] for r, d in speakers.items()}, ensure_ascii=False))

    log.info("loading whisper (%s)…", WHISPER_REPO)
    transcriber = Transcriber()

    token = access_token()
    lib = MIoTCamera(cloud_server=CLOUD_SERVER, access_token=token, frame_interval=500)
    await lib.init_async(frame_interval=500)
    loop = asyncio.get_running_loop()
    rooms: dict[str, Room] = {}

    for did in camera_dids:
        cam = by_did[did]
        room = rooms[did] = Room(cam, speakers.get(cam.get("room_name")), transcriber)
        info = dict(cam, channel_count=1, camera_status=MIoTCameraStatus.DISCONNECTED)
        inst = await lib.create_camera_async(MIoTCameraInfo.model_construct(**info))

        async def on_status(_did, status, _name=room.name):
            log.info("[%s] camera %s", _name, MIoTCameraStatus(status).name)

        await inst.register_status_changed_async(on_status)
        await inst.register_decode_jpg_async(room.on_jpg)
        await inst.register_decode_pcm_async(room.on_pcm)
        await inst.start_async(qualities=MIoTCameraVideoQuality.HIGH, enable_audio=True,
                               enable_reconnect=True)
        log.info("[%s] streaming %s; speaker: %s", room.name, cam["name"],
                 room.speaker["name"] if room.speaker else "none")

    def on_connect(c, *_):
        for did in camera_dids:
            c.subscribe(f"device/{did}/up/event_occured/#", qos=1)
        log.info("mqtt connected, subscribed to %d camera(s)", len(camera_dids))

    def on_message(_c, _u, msg):
        parts = msg.topic.split("/")
        did, key = parts[1], "/".join(parts[-2:])
        if key in EVENTS and did in rooms:
            log.info("[%s] event %s", rooms[did].name, EVENTS[key])
            loop.call_soon_threadsafe(rooms[did].on_event, EVENTS[key])

    mq = Client(callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=f"miloco:{uuid.uuid4().hex}", protocol=MQTTv5)
    mq.username_pw_set(OAUTH2_CLIENT_ID, token)
    mq.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
    mq.on_connect, mq.on_message = on_connect, on_message
    mq.reconnect_delay_set(1, 60)
    mq.connect(f"{CLOUD_SERVER}-{MIHOME_MQTT_BROKER_HOST_SUFFIX}", MIHOME_MQTT_PORT,
               keepalive=60, clean_start=True)
    mq.loop_start()

    log.info("room agent running")
    tick = 0
    while True:  # health line every minute; refresh the access token every 30 min
        await asyncio.sleep(60)
        tick += 1
        for r in rooms.values():
            secs = sum(p.size for _, p in r.audio) / 16000
            log.info("[%s] buffer: %d frames, %.1fs audio, conversation %s", r.name, len(r.frames),
                     secs, "open" if time.time() < r.conversation_until else "closed")
        if tick % 30:
            continue
        try:
            new = access_token()
            if new != token:
                token = new
                mq.username_pw_set(OAUTH2_CLIENT_ID, token)
                await lib.update_access_token_async(token)
                log.info("access token refreshed")
        except Exception as e:  # noqa: BLE001
            log.warning("token refresh check failed: %s", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cameras", nargs="+", help="camera dids to watch")
    ap.add_argument("--greet-on-entry", action="store_true",
                    help="also post an 'entered' message when someone enters an empty room")
    ap.add_argument("--language", default="zh",
                    help="Whisper language code, or 'auto' to detect (default: zh)")
    ap.add_argument("--reply-language", default="Chinese",
                    help="language Claude is asked to answer in (default: Chinese)")
    args = ap.parse_args()
    GREET_ON_ENTRY = args.greet_on_entry
    WHISPER_LANGUAGE = None if args.language == "auto" else args.language
    REPLY_LANGUAGE = args.reply_language
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for noisy in ("miot", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(main(args.cameras))

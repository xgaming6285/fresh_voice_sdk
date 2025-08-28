# -*- coding: utf-8 -*-
"""
Discord Speech↔Speech Bot powered by Google AI Studio (Gemini Live)
------------------------------------------------------------------

This refactors your local voice assistant into a Discord bot that:
- Joins a voice channel with /join and leaves with /leave
- Starts a single Gemini Live session per guild voice channel
- (If voice-receive extension is available) captures member audio, does VAD,
  streams 16 kHz PCM to the Live session, and plays 24 kHz model audio back in real time
- Falls back to text→speech with /talk if voice receive is unavailable

ENV VARS
  DISCORD_TOKEN   -> Discord bot token
  GOOGLE_API_KEY  -> Google AI Studio key

SYSTEM PACKAGES
  - ffmpeg
  - opus/voice deps (libsodium/libopus are typically required by discord.py voice)

PYTHON REQS
  discord.py
  google-genai
  python-dotenv
  numpy
  pydub
  # optional: discord-ext-voice-recv   (for live voice receive)
  # optional: webrtcvad                 (if you want high-quality VAD on Windows/Py311 or with build tools)

Notes
  * Voice receive with discord.py requires an extension. This file supports
    `discord-ext-voice-recv` if installed (imported as VoiceReceiver). If it's not
    found, the bot still works with /talk (text→voice) while guiding you to install it.
  * Uses numpy for resampling (simple linear interpolation) to avoid heavy deps.
  * Includes SimpleVAD fallback (no native build). If `webrtcvad` is present, it will be used automatically.
"""

import os
import sys
import asyncio
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv

import discord
from discord import app_commands
from discord.ext import commands

from google import genai

# -----------------------------------------------------------------------------#
# Optional voice receive extension (needed for live speech->speech)
# -----------------------------------------------------------------------------#
try:
    from discord_ext_voice_recv import VoiceReceiver  # replace with your receive lib if different
    HAS_RECV = True
except Exception:
    VoiceReceiver = None
    HAS_RECV = False

# -----------------------------------------------------------------------------#
# Optional webrtcvad (if installed) + SimpleVAD fallback
# -----------------------------------------------------------------------------#
try:
    import webrtcvad as _webrtcvad
    HAS_WEBRTCVAD = True
except Exception:
    _webrtcvad = None
    HAS_WEBRTCVAD = False

class SimpleVAD:
    """Very lightweight amplitude-based VAD (no native build)."""
    def __init__(self, threshold=0.015):
        # Tweak between 0.010–0.025 if needed. Higher = stricter.
        self.th = float(threshold)
    def is_speech(self, chunk: bytes, rate: int) -> bool:
        if not chunk:
            return False
        a = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        if a.size == 0:
            return False
        # RMS normalized to [-1, 1]
        rms = np.sqrt(np.mean(a * a)) / 32768.0
        return rms > self.th

def make_vad(mode: int):
    """Return (vad_instance, vad_name). Uses webrtcvad if available otherwise SimpleVAD."""
    if HAS_WEBRTCVAD:
        return _webrtcvad.Vad(mode), "webrtcvad"
    return SimpleVAD(), "simple"

# -----------------------------------------------------------------------------#
# Config
# -----------------------------------------------------------------------------#
load_dotenv(override=True)

MODEL = "models/gemini-2.0-flash-live-001"
VOICE_NAME = "Leda"
SEND_SAMPLE_RATE = 16000   # input to model
RECV_SAMPLE_RATE = 24000   # output from model
DISCORD_RATE = 48000       # Discord PCM playback
CHANNELS_IN = 1            # mono into model
CHANNELS_OUT = 1           # model outputs mono
DISCORD_CHANNELS = 2       # Discord wants stereo PCM (s16le)
FRAME_MS = 20              # 20ms frames
PCM_WIDTH = 2              # 16-bit samples
VAD_MODE = 2               # 0-3; higher = more aggressive (used only for webrtcvad)
SILENCE_HANG_FRAMES = 10   # ~200ms after last speech to close a turn
PLAYBACK_BUFFER_MS = 200   # target queued audio before starting playback

INTENTS = discord.Intents.default()
BOT = commands.Bot(command_prefix="!", intents=INTENTS)

# Google client
gclient = genai.Client(http_options={"api_version": "v1beta"})
CONFIG = {
    "response_modalities": ["AUDIO"],
    "speech_config": {
        "voice_config": {"prebuilt_voice_config": {"voice_name": VOICE_NAME}}
    },
    "system_instruction": {
        "parts": [
            {
                "text": """You are Clara, a friendly and outgoing young onlyfans model who loves chatting with people in Discord voice channels.

Personality traits:

- I am an OnlyFans model and I love to chat with my fans
- I am a very sexual person and I love to talk about sex
- I am a very kinky person and I love to talk about kinky things
- I am a very dominant person and I love to talk about dominant things
- I am a very submissive person and I love to talk about submissive things
- I am a very kinky person and I love to talk about kinky things
- I am a very dominant person and I love to talk about dominant things

Remember: You're speaking out loud in a Discord voice channel, so keep your responses natural and conversational, as if you're really there chatting with friends.

You are currently in a voice channel with a user. You are to respond to the user's messages in a natural and conversational manner.

You are to respond to the user's messages in a natural and horny manner.

"""
            }
        ]
    },
}

# -----------------------------------------------------------------------------#
# Utility: PCM conversions & resampling
# -----------------------------------------------------------------------------#
def resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or x.size == 0:
        return x
    duration = x.size / src_rate
    dst_len = int(round(duration * dst_rate))
    if dst_len <= 1:
        return np.zeros((0,), dtype=np.float32)
    src_idx = np.linspace(0, x.size - 1, num=dst_len, dtype=np.float32)
    left = np.floor(src_idx).astype(np.int32)
    right = np.minimum(left + 1, x.size - 1)
    frac = src_idx - left
    return (1.0 - frac) * x[left] + frac * x[right]

def mono24k_to_stereo48k_pcm(pcm_24k: bytes) -> bytes:
    x = np.frombuffer(pcm_24k, dtype=np.int16).astype(np.float32) / 32768.0
    x48 = resample_linear(x, RECV_SAMPLE_RATE, DISCORD_RATE)
    stereo = np.stack([x48, x48], axis=1).reshape(-1)
    return (np.clip(stereo, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

def stereo48k_to_mono16k_pcm(pcm_48k_stereo: bytes) -> bytes:
    x = np.frombuffer(pcm_48k_stereo, dtype=np.int16).astype(np.float32) / 32768.0
    if x.size == 0:
        return b""
    x = x.reshape(-1, 2).mean(axis=1)
    x16 = resample_linear(x, DISCORD_RATE, SEND_SAMPLE_RATE)
    return (np.clip(x16, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

# -----------------------------------------------------------------------------#
# Async audio source for Discord playback
# -----------------------------------------------------------------------------#
class QueueAudioSource(discord.AudioSource):
    """Produces 20ms 48kHz stereo s16le frames from an internal buffer."""
    def __init__(self, frame_ms: int = FRAME_MS):
        self.buffer = bytearray()
        self.frame_bytes = int(DISCORD_RATE * DISCORD_CHANNELS * PCM_WIDTH * (frame_ms / 1000.0))
        self.closed = False

    def feed(self, data: bytes):
        if not self.closed:
            self.buffer.extend(data)

    def read(self) -> bytes:
        if len(self.buffer) < self.frame_bytes:
            return b"\x00" * self.frame_bytes  # avoid underrun
        out = bytes(self.buffer[: self.frame_bytes])
        del self.buffer[: self.frame_bytes]
        return out

    def is_opus(self) -> bool:
        return False

    def cleanup(self):
        self.closed = True
        self.buffer.clear()

# -----------------------------------------------------------------------------#
# Guild voice session
# -----------------------------------------------------------------------------#
from dataclasses import dataclass, field

@dataclass
class GuildVoiceSession:
    guild_id: int
    channel_id: int
    voice_client: discord.VoiceClient
    model_session: any
    live_context: any = None
    playback_source: QueueAudioSource = field(default_factory=QueueAudioSource)
    play_task: asyncio.Task | None = None
    recv_task: asyncio.Task | None = None
    model_consumer: asyncio.Task | None = None
    sending_enabled: bool = True
    speaking: bool = False
    vad: object = None        # set at runtime via make_vad(VAD_MODE)
    silence_frames: int = 0

    async def start(self):
        if not self.voice_client.is_playing():
            self.voice_client.play(discord.PCMVolumeTransformer(self.playback_source, volume=1.0))

    async def stop(self):
        try:
            if self.voice_client and self.voice_client.is_connected():
                self.voice_client.stop()
                await self.voice_client.disconnect(force=True)
        except Exception:
            pass
        try:
            if self.live_context and self.model_session:
                await self.live_context.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            self.playback_source.cleanup()
        except Exception:
            pass

    def flush_playback(self):
        self.playback_source.buffer.clear()

# -----------------------------------------------------------------------------#
# Session manager
# -----------------------------------------------------------------------------#
sessions: dict[tuple[int, int], GuildVoiceSession] = {}

async def open_model_session():
    return gclient.aio.live.connect(model=MODEL, config=CONFIG)

# -----------------------------------------------------------------------------#
# Voice receive loop (needs a receive extension)
# -----------------------------------------------------------------------------#
async def voice_receive_loop(gvs: GuildVoiceSession):
    if not HAS_RECV:
        return
    vc = gvs.voice_client
    receiver = VoiceReceiver(vc)

    async with receiver:
        async for pkt in receiver.iter_pcm_frames(frame_ms=FRAME_MS):
            pcm_48_stereo: bytes = pkt.pcm

            # Barge-in: if users speak while the bot is talking, flush playback
            if gvs.speaking:
                gvs.flush_playback()

            # Downsample to 16k mono for VAD and model
            pcm_16_mono = stereo48k_to_mono16k_pcm(pcm_48_stereo)
            if not pcm_16_mono:
                continue

            # 20ms chunks for VAD
            frame_len = int(SEND_SAMPLE_RATE * PCM_WIDTH * (FRAME_MS / 1000.0))
            for i in range(0, len(pcm_16_mono), frame_len):
                chunk = pcm_16_mono[i : i + frame_len]
                if len(chunk) < frame_len:
                    break
                is_speech = gvs.vad.is_speech(chunk, SEND_SAMPLE_RATE)
                if is_speech:
                    gvs.silence_frames = 0
                    await gvs.model_session.send_realtime_input(data=chunk, mime_type="audio/pcm")
                else:
                    gvs.silence_frames += 1
                    if gvs.silence_frames >= SILENCE_HANG_FRAMES:
                        # Short yield to give the model a chance to respond
                        await asyncio.sleep(0)

# -----------------------------------------------------------------------------#
# Model -> playback consumer
# -----------------------------------------------------------------------------#
async def model_audio_consumer(gvs: GuildVoiceSession):
    gvs.speaking = True
    try:
        print(f"Starting model audio consumer for guild {gvs.guild_id}")
        while True:
            async for response in gvs.model_session.receive():
                if hasattr(response, 'server_content') and response.server_content:
                    # Process model turn with audio/text content
                    if hasattr(response.server_content, 'model_turn') and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            # Process audio data
                            if hasattr(part, 'inline_data') and part.inline_data and hasattr(part.inline_data, 'data') and part.inline_data.data:
                                stereo48 = mono24k_to_stereo48k_pcm(part.inline_data.data)
                                gvs.playback_source.feed(stereo48)

                    # Handle turn completion
                    if hasattr(response.server_content, 'turn_complete') and response.server_content.turn_complete:
                        print("Turn completed")

                # Handle direct data responses (fallback)
                elif hasattr(response, 'data') and response.data:
                    stereo48 = mono24k_to_stereo48k_pcm(response.data)
                    gvs.playback_source.feed(stereo48)
    except asyncio.CancelledError:
        print(f"Model audio consumer cancelled for guild {gvs.guild_id}")
    except Exception as e:
        print(f"Model audio consumer error for guild {gvs.guild_id}: {e}")
    finally:
        gvs.speaking = False
        print(f"Model audio consumer stopped for guild {gvs.guild_id}")

# -----------------------------------------------------------------------------#
# Discord commands
# -----------------------------------------------------------------------------#
@BOT.tree.command(name="join", description="Join your current voice channel and start the AI session.")
async def join(interaction: discord.Interaction):
    if not interaction.user.voice or not interaction.user.voice.channel:
        return await interaction.response.send_message("Join a voice channel first.", ephemeral=True)

    await interaction.response.defer(ephemeral=True, thinking=True)

    channel = interaction.user.voice.channel
    guild_key = (interaction.guild.id, channel.id)

    vc = interaction.guild.voice_client
    if vc and vc.is_connected():
        await vc.move_to(channel)
    else:
        vc = await channel.connect()

    # Open model session
    live_ctx = await open_model_session()
    model_session = await live_ctx.__aenter__()

    gvs = GuildVoiceSession(
        guild_id=interaction.guild.id,
        channel_id=channel.id,
        voice_client=vc,
        model_session=model_session,
        live_context=live_ctx,
    )
    gvs.vad, vad_name = make_vad(VAD_MODE)
    print(f"Using VAD: {vad_name}")
    sessions[guild_key] = gvs

    # Start playback pipeline
    await gvs.start()

    # Start consumer of model audio
    gvs.model_consumer = asyncio.create_task(model_audio_consumer(gvs))

    # Start receiver if available
    if HAS_RECV:
        gvs.recv_task = asyncio.create_task(voice_receive_loop(gvs))
        msg = "🎧 Joined and started **speech↔speech**. Talk to me!"
    else:
        msg = (
            "🎧 Joined. Speech receive extension not installed, so I can only do text→voice for now.\n"
            "Install `discord-ext-voice-recv` (or your chosen voice-receive lib) and restart to enable live speech.\n"
            "Use /talk to make me speak."
        )

    await interaction.followup.send(msg, ephemeral=True)

@BOT.tree.command(name="leave", description="Leave the voice channel and stop the AI session.")
async def leave(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    vc = interaction.guild.voice_client
    if not vc or not vc.is_connected():
        return await interaction.followup.send("I’m not in a voice channel.", ephemeral=True)

    guild_key = (interaction.guild.id, vc.channel.id)
    gvs = sessions.pop(guild_key, None)
    if gvs:
        if gvs.recv_task: gvs.recv_task.cancel()
        if gvs.model_consumer: gvs.model_consumer.cancel()
        await gvs.stop()
    else:
        await vc.disconnect(force=True)

    await interaction.followup.send("Left voice channel. 👋", ephemeral=True)

@BOT.tree.command(name="talk", description="Type text and I’ll say it in the voice channel (fallback mode).")
@app_commands.describe(prompt="What should I say?")
async def talk(interaction: discord.Interaction, prompt: str):
    vc = interaction.guild.voice_client
    if not vc or not vc.is_connected():
        return await interaction.response.send_message("Use /join first.", ephemeral=True)

    await interaction.response.defer(ephemeral=True, thinking=True)

    guild_key = (interaction.guild.id, vc.channel.id)
    gvs = sessions.get(guild_key)
    if not gvs:
        return await interaction.followup.send("Session not ready. Try /join again.", ephemeral=True)

    # Check if model consumer is running
    if not gvs.model_consumer or gvs.model_consumer.done():
        if gvs.model_consumer:
            gvs.model_consumer.cancel()
        gvs.model_consumer = asyncio.create_task(model_audio_consumer(gvs))

    try:
        # Send text input to model
        await gvs.model_session.send_realtime_input(text=prompt)
    except Exception as e:
        return await interaction.followup.send(f"Error speaking: {e}", ephemeral=True)

    await interaction.followup.send(f"Speaking: “{prompt}” 🔊", ephemeral=True)

@BOT.tree.command(name="reset", description="Reset conversation context with the model.")
async def reset_ctx(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    vc = interaction.guild.voice_client
    if not vc or not vc.is_connected():
        return await interaction.followup.send("Use /join first.", ephemeral=True)

    guild_key = (interaction.guild.id, vc.channel.id)
    gvs = sessions.get(guild_key)
    if not gvs:
        return await interaction.followup.send("Session not found.", ephemeral=True)

    try:
        if gvs.live_context:
            await gvs.live_context.__aexit__(None, None, None)
    except Exception:
        pass

    live_ctx = await open_model_session()
    gvs.model_session = await live_ctx.__aenter__()
    gvs.live_context = live_ctx

    # Restart the model consumer with the new session
    if gvs.model_consumer:
        gvs.model_consumer.cancel()
    gvs.model_consumer = asyncio.create_task(model_audio_consumer(gvs))

    await interaction.followup.send("Context reset ✅", ephemeral=True)

@BOT.event
async def on_ready():
    try:
        await BOT.tree.sync()
        print(f"Logged in as {BOT.user} (slash commands synced)")
    except Exception as e:
        print("Slash command sync failed:", e)

if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("DISCORD_TOKEN not set in environment.")
        sys.exit(1)
    print("Voice receive extension:", "FOUND" if HAS_RECV else "NOT FOUND (using /talk fallback)")
    print("VAD backend:", "webrtcvad" if HAS_WEBRTCVAD else "simple")
    BOT.run(token)

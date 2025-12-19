#!/usr/bin/env python3
"""
AI Necklace Realtime - Raspberry Pi 5 リアルタイム音声AIアシスタント

OpenAI Realtime APIを使用したリアルタイム双方向音声対話システム。
ボタンを押している間だけ音声入力し、低レイテンシで応答を得る。

ボタン操作:
- ボタンを押す → 音声入力開始（AIの応答中なら割り込み）
- ボタンを離す → 音声入力終了 → AI応答開始
"""

import os
import sys
import json
import base64
import asyncio
import threading
import signal
import time
from datetime import datetime

import numpy as np
import pyaudio
import websockets
from dotenv import load_dotenv

# GPIOライブラリ
try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("警告: gpiozeroが使用できません。キーボード操作モードで動作します。")

# systemdで実行時にprint出力をリアルタイムで表示するため
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 環境変数の読み込み
load_dotenv()

# 設定
CONFIG = {
    # Realtime API設定
    "model": "gpt-4o-realtime-preview-2024-12-17",
    "voice": "shimmer",  # alloy, echo, shimmer, etc.

    # オーディオ設定（Realtime APIは24kHz, 16bit PCM, モノラル）
    "sample_rate": 24000,
    "channels": 1,
    "chunk_size": 1024,  # 約42ms @ 24kHz
    "output_sample_rate": 48000,  # USBスピーカー用にリサンプリング

    # デバイス設定
    "input_device_index": None,
    "output_device_index": None,

    # GPIO設定
    "button_pin": 5,
    "use_button": True,

    # システムプロンプト
    "instructions": """あなたは親切なAIアシスタントです。
ユーザーの質問に簡潔に答えてください。
日本語で回答してください。
音声での会話なので、1-2文程度の短い応答を心がけてください。
""",
}

# グローバル変数
running = True
audio = None
button = None
ws_connection = None
is_speaking = False  # AIが話している間はTrue
is_recording = False  # ユーザーが録音中はTrue
audio_output_buffer = []
audio_output_lock = threading.Lock()


def signal_handler(sig, frame):
    """Ctrl+C で終了"""
    global running
    print("\n終了します...")
    running = False


def find_audio_device(p, device_type="input"):
    """オーディオデバイスを自動検出"""
    target_names = ["USB PnP Sound", "USB Audio", "USB PnP Audio"]

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get("name", "")

        if device_type == "input" and info.get("maxInputChannels", 0) > 0:
            for target in target_names:
                if target in name:
                    print(f"入力デバイス検出: [{i}] {name}")
                    return i
        elif device_type == "output" and info.get("maxOutputChannels", 0) > 0:
            for target in target_names:
                if target in name:
                    print(f"出力デバイス検出: [{i}] {name}")
                    return i

    if device_type == "input":
        return p.get_default_input_device_info()["index"]
    else:
        return p.get_default_output_device_info()["index"]


def resample_audio(audio_data, from_rate, to_rate):
    """オーディオをリサンプリング"""
    if from_rate == to_rate:
        return audio_data

    # int16のバイトデータをnumpy配列に変換
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # リサンプリング（線形補間）
    original_length = len(audio_array)
    target_length = int(original_length * to_rate / from_rate)
    indices = np.linspace(0, original_length - 1, target_length)
    resampled = np.interp(indices, np.arange(original_length), audio_array)

    return resampled.astype(np.int16).tobytes()


class RealtimeAudioHandler:
    """リアルタイム音声処理ハンドラ"""

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.is_recording = False
        self.is_playing = False
        self.output_buffer = []
        self.output_lock = threading.Lock()

    def start_input_stream(self):
        """マイク入力ストリームを開始"""
        input_device = CONFIG["input_device_index"]
        if input_device is None:
            input_device = find_audio_device(self.audio, "input")

        self.input_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=CONFIG["channels"],
            rate=CONFIG["sample_rate"],
            input=True,
            input_device_index=input_device,
            frames_per_buffer=CONFIG["chunk_size"]
        )
        self.is_recording = True
        print("🎤 マイク入力開始")

    def stop_input_stream(self):
        """マイク入力ストリームを停止"""
        if self.input_stream:
            self.is_recording = False
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
            print("🎤 マイク入力停止")

    def read_audio_chunk(self):
        """マイクから音声チャンクを読み取り"""
        if self.input_stream and self.is_recording:
            try:
                data = self.input_stream.read(CONFIG["chunk_size"], exception_on_overflow=False)
                return data
            except Exception as e:
                print(f"音声読み取りエラー: {e}")
        return None

    def start_output_stream(self):
        """スピーカー出力ストリームを開始"""
        output_device = CONFIG["output_device_index"]
        if output_device is None:
            output_device = find_audio_device(self.audio, "output")

        self.output_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=CONFIG["channels"],
            rate=CONFIG["output_sample_rate"],
            output=True,
            output_device_index=output_device,
            frames_per_buffer=CONFIG["chunk_size"] * 2
        )
        self.is_playing = True
        print("🔊 スピーカー出力開始")

    def stop_output_stream(self):
        """スピーカー出力ストリームを停止"""
        if self.output_stream:
            self.is_playing = False
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
            print("🔊 スピーカー出力停止")

    def play_audio_chunk(self, audio_data):
        """音声チャンクを再生"""
        if self.output_stream and self.is_playing:
            try:
                # 24kHz → 48kHz にリサンプリング
                resampled = resample_audio(
                    audio_data,
                    CONFIG["sample_rate"],
                    CONFIG["output_sample_rate"]
                )
                self.output_stream.write(resampled)
            except Exception as e:
                print(f"音声再生エラー: {e}")

    def add_to_output_buffer(self, audio_data):
        """出力バッファに音声を追加"""
        with self.output_lock:
            self.output_buffer.append(audio_data)

    def clear_output_buffer(self):
        """出力バッファをクリア（割り込み時）"""
        with self.output_lock:
            self.output_buffer.clear()

    def cleanup(self):
        """リソースを解放"""
        self.stop_input_stream()
        self.stop_output_stream()
        if self.audio:
            self.audio.terminate()


class RealtimeClient:
    """OpenAI Realtime APIクライアント"""

    def __init__(self, audio_handler: RealtimeAudioHandler):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY が設定されていません")

        self.audio_handler = audio_handler
        self.ws = None
        self.is_connected = False
        self.is_responding = False
        self.current_response_id = None

    async def connect(self):
        """WebSocket接続を確立"""
        url = f"wss://api.openai.com/v1/realtime?model={CONFIG['model']}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        print(f"🔗 Realtime APIに接続中... ({CONFIG['model']})")

        self.ws = await websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
        )
        self.is_connected = True
        print("✅ Realtime API接続完了")

        # セッション設定を送信
        await self.configure_session()

    async def configure_session(self):
        """セッション設定を送信"""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": CONFIG["instructions"],
                "voice": CONFIG["voice"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": None,  # 手動モード（ボタン操作）
            }
        }

        await self.ws.send(json.dumps(session_config))
        print("📝 セッション設定完了（手動ターン検出モード）")

    async def send_audio_chunk(self, audio_data):
        """音声チャンクを送信"""
        if not self.is_connected or not self.ws:
            return

        encoded = base64.b64encode(audio_data).decode("utf-8")
        message = {
            "type": "input_audio_buffer.append",
            "audio": encoded
        }
        await self.ws.send(json.dumps(message))

    async def commit_audio(self):
        """音声入力を確定して応答を要求"""
        if not self.is_connected or not self.ws:
            return

        # 音声バッファをコミット
        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # 応答を生成
        await self.ws.send(json.dumps({"type": "response.create"}))
        print("📤 音声送信完了、応答待ち...")

    async def cancel_response(self):
        """現在の応答をキャンセル（割り込み）"""
        if self.is_responding and self.ws:
            await self.ws.send(json.dumps({"type": "response.cancel"}))
            self.audio_handler.clear_output_buffer()
            print("⚡ 応答をキャンセル（割り込み）")

    async def clear_input_buffer(self):
        """入力バッファをクリア"""
        if self.ws:
            await self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

    async def receive_messages(self):
        """メッセージを受信して処理"""
        try:
            async for message in self.ws:
                if not running:
                    break

                event = json.loads(message)
                await self.handle_event(event)

        except websockets.exceptions.ConnectionClosed:
            print("⚠️ WebSocket接続が閉じられました")
            self.is_connected = False
        except Exception as e:
            print(f"⚠️ 受信エラー: {e}")

    async def handle_event(self, event):
        """イベントを処理"""
        event_type = event.get("type", "")

        if event_type == "session.created":
            print("🎉 セッション作成完了")

        elif event_type == "session.updated":
            print("📝 セッション更新完了")

        elif event_type == "response.created":
            self.is_responding = True
            self.current_response_id = event.get("response", {}).get("id")

        elif event_type == "response.audio.delta":
            # 音声データを受信
            audio_b64 = event.get("delta", "")
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)
                self.audio_handler.play_audio_chunk(audio_data)

        elif event_type == "response.audio_transcript.delta":
            # テキストトランスクリプト（デバッグ用）
            text = event.get("delta", "")
            if text:
                print(f"[AI] {text}", end="", flush=True)

        elif event_type == "response.audio_transcript.done":
            print()  # 改行

        elif event_type == "response.done":
            self.is_responding = False
            self.current_response_id = None
            print("✅ 応答完了")

        elif event_type == "input_audio_buffer.speech_started":
            print("🎤 音声検出開始")

        elif event_type == "input_audio_buffer.speech_stopped":
            print("🎤 音声検出終了")

        elif event_type == "conversation.item.input_audio_transcription.completed":
            # ユーザーの発話のトランスクリプト
            transcript = event.get("transcript", "")
            if transcript:
                print(f"[あなた] {transcript}")

        elif event_type == "error":
            error = event.get("error", {})
            print(f"❌ エラー: {error.get('message', 'Unknown error')}")

        elif event_type == "rate_limits.updated":
            pass  # レート制限の更新は無視

        else:
            # デバッグ用：未処理のイベント
            # print(f"📩 イベント: {event_type}")
            pass

    async def disconnect(self):
        """接続を切断"""
        if self.ws:
            await self.ws.close()
            self.is_connected = False
            print("🔌 Realtime API切断")


async def audio_input_loop(client: RealtimeClient, audio_handler: RealtimeAudioHandler):
    """音声入力ループ（ボタン押下中のみ送信）"""
    global running, button, is_recording

    while running:
        # ボタン操作を確認
        if CONFIG["use_button"] and button:
            if button.is_pressed:
                if not is_recording:
                    # 録音開始
                    is_recording = True

                    # AIが話していたら割り込み
                    if client.is_responding:
                        await client.cancel_response()

                    # 入力バッファをクリア
                    await client.clear_input_buffer()

                    audio_handler.start_input_stream()

                # 音声を読み取って送信
                chunk = audio_handler.read_audio_chunk()
                if chunk:
                    await client.send_audio_chunk(chunk)

            else:
                if is_recording:
                    # 録音終了
                    is_recording = False
                    audio_handler.stop_input_stream()

                    # 音声をコミットして応答を要求
                    await client.commit_audio()
        else:
            # ボタンなしモード（キーボード入力で代替）
            pass

        await asyncio.sleep(0.01)  # 10ms間隔


async def main_async():
    """非同期メインループ"""
    global running, button

    # オーディオハンドラを初期化
    audio_handler = RealtimeAudioHandler()

    # 出力ストリームを開始
    audio_handler.start_output_stream()

    # Realtimeクライアントを初期化
    client = RealtimeClient(audio_handler)

    try:
        # 接続
        await client.connect()

        # 受信タスクと入力タスクを並行実行
        receive_task = asyncio.create_task(client.receive_messages())
        input_task = asyncio.create_task(audio_input_loop(client, audio_handler))

        print("\n" + "=" * 50)
        print("AI Necklace Realtime 起動")
        print("=" * 50)
        if CONFIG["use_button"]:
            print(f"操作方法: GPIO{CONFIG['button_pin']}のボタンを押している間話す")
        print("Ctrl+C で終了")
        print("=" * 50)
        print("\n--- ボタンを押して話しかけてください ---\n")

        # 終了を待機
        while running:
            await asyncio.sleep(0.1)

        # タスクをキャンセル
        receive_task.cancel()
        input_task.cancel()

        try:
            await receive_task
        except asyncio.CancelledError:
            pass

        try:
            await input_task
        except asyncio.CancelledError:
            pass

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # クリーンアップ
        await client.disconnect()
        audio_handler.cleanup()


def main():
    """メインエントリーポイント"""
    global running, button

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # API キー確認
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("エラー: OPENAI_API_KEY が設定されていません")
        print(".env ファイルに OPENAI_API_KEY=sk-... を設定してください")
        sys.exit(1)

    # ボタン初期化
    if CONFIG["use_button"] and GPIO_AVAILABLE:
        try:
            button = Button(CONFIG["button_pin"], pull_up=True, bounce_time=0.1)
            print(f"ボタン初期化完了: GPIO{CONFIG['button_pin']}")
        except Exception as e:
            print(f"ボタン初期化エラー: {e}")
            print("ボタンなしモードで動作します")
            button = None
            CONFIG["use_button"] = False
    else:
        button = None
        if CONFIG["use_button"]:
            print("GPIOが使用できないため、ボタンなしモードで動作します")
            CONFIG["use_button"] = False

    # 非同期メインループを実行
    asyncio.run(main_async())

    print("終了しました")


if __name__ == "__main__":
    main()

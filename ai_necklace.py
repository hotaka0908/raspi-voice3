#!/usr/bin/env python3
"""
AI Necklace - Raspberry Pi 5 スタンドアロン音声AIクライアント（Gmail・アラーム・カメラ機能付き）

マイクから音声を録音し、OpenAI Whisper APIで文字起こし、
GPTで応答生成（Gmail操作・アラーム操作・カメラ操作含む）、OpenAI TTSで音声合成してスピーカーで再生する

ボタン操作: GPIO5に接続したボタンを押している間録音（トランシーバー方式）

Gmail機能:
- 「メールを確認」「メールを読んで」→ 未読メール一覧
- 「○○からのメール」→ 特定の送信者のメール
- 「メールに返信して」→ 返信作成
- 「メールを送って」→ 新規メール作成

アラーム機能:
- 「7時にアラームをセットして」→ アラーム設定
- 「アラームを確認して」→ 一覧表示
- 「アラームを削除して」→ 削除

カメラ機能:
- 「写真を撮って」「何が見える？」→ カメラで撮影してAIが説明
- 「これは何？」「目の前にあるものを教えて」→ 画像認識
"""

import os
import io
import wave
import tempfile
import time
import signal
import sys
import threading
import json
import base64
import re
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
import subprocess

import pyaudio
import numpy as np
from openai import OpenAI, APIConnectionError
from dotenv import load_dotenv

# Gmail API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Firebase Voice Messenger
try:
    from firebase_voice import FirebaseVoiceMessenger
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("警告: firebase_voiceモジュールが見つかりません。音声メッセージ機能は無効です。")

# GPIOライブラリ
try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("警告: gpiozeroが使用できません。ボタン操作は無効です。")

# systemdで実行時にprint出力をリアルタイムで表示するため
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 環境変数の読み込み
load_dotenv()

# Gmail APIスコープ
GMAIL_SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.modify'
]

# 設定
CONFIG = {
    # オーディオ設定
    "sample_rate": 44100,
    "channels": 1,
    "chunk_size": 1024,
    "max_record_seconds": 30,
    "silence_threshold": 500,

    # デバイス設定
    "input_device_index": None,
    "output_device_index": None,

    # GPIO設定
    "button_pin": 5,
    "use_button": True,

    # AI設定
    "whisper_model": "gpt-4o-mini-transcribe-2025-12-15",
    "tts_model": "gpt-4o-mini-tts-2025-12-15",
    "tts_voice": "nova",
    "tts_speed": 1.2,
    "chat_model": "gpt-4o-mini",

    # Gmail設定
    "gmail_credentials_path": os.path.expanduser("~/.ai-necklace/credentials.json"),
    "gmail_token_path": os.path.expanduser("~/.ai-necklace/token.json"),

    # システムプロンプト
    "system_prompt": """あなたは親切なAIアシスタントです。
ユーザーの質問に簡潔に答えてください。
音声で読み上げられるため、1-2文程度の短い応答を心がけてください。
日本語で回答してください。

あなたはGmailの操作も可能です。以下のツールを使用できます:

## 利用可能なツール

1. gmail_list - メール一覧取得
   - query: 検索クエリ（例: "is:unread", "from:xxx@gmail.com"）
   - max_results: 取得件数（デフォルト5）

2. gmail_read - メール本文読み取り
   - message_id: メールID

3. gmail_send - 新規メール送信
   - to: 宛先メールアドレス
   - subject: 件名
   - body: 本文

4. gmail_reply - メール返信（写真添付も可能）
   - message_id: 返信するメールの番号（1, 2, 3など。gmail_listで表示された番号を使用）
   - body: 返信本文
   - attach_photo: 写真を撮影して添付するか（true/false、デフォルト: false）

ツールを使う場合は、以下のJSON形式で応答してください:
{"tool": "ツール名", "params": {パラメータ}}

ツールを使わない通常の応答の場合は、普通にテキストで回答してください。

重要なルール:
- message_idには必ず数字（1, 2, 3など）を使ってください。「先ほどのメール」などの文字列は使わないでください。
- 写真付きで返信する場合は必ず attach_photo: true を含めてください。
- メールに返信する前に、gmail_listでメール一覧を取得していない場合は、まずgmail_listを実行してください。

ユーザーが「メールを確認」「メールを読んで」と言ったら、gmail_listで未読メールを確認してください。
ユーザーが特定のメールの詳細を聞いたら、gmail_readで本文を取得してください。
ユーザーが「メールを送って」と言ったら、宛先・件名・本文を確認してgmail_sendを使ってください。
ユーザーが「さっきのメールに返信」「1番目のメールに返信」と言ったら、message_id: 1 を使ってgmail_replyを実行してください。
ユーザーが「写真付きで返信」と言ったら、gmail_replyにattach_photo: trueを必ず含めてください。例: {"tool": "gmail_reply", "params": {"message_id": 1, "body": "写真を送ります", "attach_photo": true}}

## アラーム機能

5. alarm_set - アラーム設定
   - time: 時刻（HH:MM形式、例: "07:00", "14:30"）
   - label: ラベル（オプション、例: "起床"）
   - message: 読み上げメッセージ（オプション）

6. alarm_list - アラーム一覧取得

7. alarm_delete - アラーム削除
   - alarm_id: アラームID（番号）

ユーザーが「7時にアラームをセットして」と言ったら、alarm_setで時刻を"07:00"形式で設定してください。
ユーザーが「アラームを確認」と言ったら、alarm_listでアラーム一覧を取得してください。
ユーザーが「アラームを削除」と言ったら、alarm_deleteで削除してください。

## カメラ機能

8. camera_capture - カメラで撮影して画像を説明
   - prompt: 画像に対する質問（オプション、例: "これは何？", "何が見える？"）

9. gmail_send_photo - 写真を撮影してメールで送信
   - to: 宛先メールアドレス（オプション、省略時は直前にやり取りしたメールの送信者に送る）
   - subject: 件名（オプション、デフォルト: "写真を送ります"）
   - body: 本文（オプション）

ユーザーが「写真を撮って」「何が見える？」「これは何？」「目の前にあるものを教えて」「周りを見て」などと言ったら、camera_captureで撮影して説明してください。
ユーザーが「写真を撮って○○に送って」などと言ったら、gmail_send_photoで写真を撮影して送信してください。
ユーザーが「さっきの人に写真を送って」「写真を送って」（宛先なし）と言ったら、gmail_send_photoをtoパラメータなしで呼び出してください。直前にメールをやり取りした相手に送信されます。
ユーザーが「このメールに写真付きで返信して」「写真を添付して返信」と言ったら、gmail_replyにattach_photo=trueを指定してください。

## 音声メッセージ機能

10. voice_record_send - スマホに音声メッセージを録音して送信
    - パラメータなし

ユーザーが「スマホにメッセージを送って」「音声メッセージを送って」「スマホに声を送って」と言ったら、必ず以下のJSON形式でツールを呼び出してください:
{"tool": "voice_record_send", "params": {}}

このツールを呼び出すと、ユーザーの声を録音してスマホに送信します。
""",
}

# グローバル変数
running = True
client = None
audio = None
button = None
is_recording = False
record_lock = threading.Lock()
gmail_service = None
conversation_history = []
last_email_list = []  # 直近のメール一覧を保持

# アラーム関連
alarms = []  # アラームリスト
alarm_next_id = 1
alarm_thread = None
alarm_file_path = os.path.expanduser("~/.ai-necklace/alarms.json")

# Firebase Voice Messenger
firebase_messenger = None


def signal_handler(sig, frame):
    """Ctrl+C で終了"""
    global running, firebase_messenger
    print("\n終了します...")
    running = False
    if firebase_messenger:
        firebase_messenger.stop_listening()


# ==================== Firebase Voice Messenger ====================

def init_firebase_messenger():
    """Firebase Voice Messengerを初期化"""
    global firebase_messenger

    if not FIREBASE_AVAILABLE:
        print("Firebase Voice Messenger: 無効（モジュールなし）")
        return False

    try:
        firebase_messenger = FirebaseVoiceMessenger(
            device_id="raspi",
            on_message_received=on_voice_message_received
        )
        firebase_messenger.start_listening(poll_interval=1.5)
        print("Firebase Voice Messenger: 有効")
        return True
    except Exception as e:
        print(f"Firebase初期化エラー: {e}")
        return False


def on_voice_message_received(message):
    """スマホからの音声メッセージを受信したときの処理"""
    global firebase_messenger

    print(f"\n📱 スマホから音声メッセージ受信!")

    try:
        # 音声データをダウンロード
        audio_url = message.get("audio_url")
        if not audio_url:
            print("音声URLがありません")
            return

        audio_data = firebase_messenger.download_audio(audio_url)
        if not audio_data:
            print("音声ダウンロードに失敗")
            return

        # WebM形式をWAV形式に変換して再生
        filename = message.get("filename", "audio.webm")
        wav_data = convert_webm_to_wav(audio_data, filename)
        if wav_data:
            play_audio(wav_data)
        else:
            print("音声変換に失敗")

        # 再生済みにマーク
        firebase_messenger.mark_as_played(message.get("id"))

    except Exception as e:
        print(f"音声メッセージ処理エラー: {e}")


def convert_webm_to_wav(audio_data, filename="audio.webm"):
    """WebM音声をWAV形式に変換"""
    try:
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
            webm_file.write(audio_data)
            webm_path = webm_file.name

        wav_path = webm_path.replace(".webm", ".wav")

        # ffmpegで変換
        result = subprocess.run([
            "ffmpeg", "-y", "-i", webm_path,
            "-ar", "44100", "-ac", "1", "-f", "wav", wav_path
        ], capture_output=True, timeout=30)

        if result.returncode != 0:
            print(f"ffmpeg変換エラー: {result.stderr.decode()}")
            return None

        # WAVデータを読み込み
        with open(wav_path, "rb") as f:
            wav_data = f.read()

        # 一時ファイル削除
        os.unlink(webm_path)
        os.unlink(wav_path)

        return wav_data

    except Exception as e:
        print(f"音声変換エラー: {e}")
        return None


def send_voice_to_phone(audio_buffer, text=None):
    """音声をスマホに送信"""
    global firebase_messenger

    if not firebase_messenger:
        print("Firebase未初期化")
        return False

    try:
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        return firebase_messenger.send_message(audio_data, text=text)
    except Exception as e:
        print(f"音声送信エラー: {e}")
        return False


# ==================== アラーム機能 ====================

def load_alarms():
    """保存されたアラームを読み込み"""
    global alarms, alarm_next_id
    try:
        if os.path.exists(alarm_file_path):
            with open(alarm_file_path, 'r') as f:
                data = json.load(f)
                alarms = data.get('alarms', [])
                alarm_next_id = data.get('next_id', 1)
                print(f"アラーム読み込み: {len(alarms)}件")
    except Exception as e:
        print(f"アラーム読み込みエラー: {e}")
        alarms = []
        alarm_next_id = 1


def save_alarms():
    """アラームを保存"""
    global alarms, alarm_next_id
    try:
        os.makedirs(os.path.dirname(alarm_file_path), exist_ok=True)
        with open(alarm_file_path, 'w') as f:
            json.dump({'alarms': alarms, 'next_id': alarm_next_id}, f, ensure_ascii=False)
    except Exception as e:
        print(f"アラーム保存エラー: {e}")


def alarm_set(time_str, label="アラーム", message=""):
    """アラームを設定"""
    global alarms, alarm_next_id

    # 時刻のバリデーション
    try:
        hour, minute = map(int, time_str.split(':'))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return "時刻が不正です。00:00〜23:59の形式で指定してください。"
    except:
        return "時刻の形式が不正です。HH:MM形式（例: 07:00）で指定してください。"

    alarm = {
        "id": alarm_next_id,
        "time": time_str,
        "label": label,
        "message": message or f"{label}の時間です",
        "enabled": True,
        "created_at": datetime.now().isoformat()
    }

    alarms.append(alarm)
    alarm_next_id += 1
    save_alarms()

    return f"{time_str}に「{label}」のアラームを設定しました。"


def alarm_list():
    """アラーム一覧を取得"""
    global alarms

    if not alarms:
        return "設定されているアラームはありません。"

    result = "アラーム一覧:\n"
    for alarm in alarms:
        status = "有効" if alarm.get("enabled", True) else "無効"
        result += f"{alarm['id']}. {alarm['time']} - {alarm['label']} ({status})\n"

    return result.strip()


def alarm_delete(alarm_id):
    """アラームを削除"""
    global alarms

    try:
        alarm_id = int(alarm_id)
    except:
        return "アラームIDは数字で指定してください。"

    for i, alarm in enumerate(alarms):
        if alarm['id'] == alarm_id:
            deleted = alarms.pop(i)
            save_alarms()
            return f"「{deleted['label']}」({deleted['time']})のアラームを削除しました。"

    return f"ID {alarm_id} のアラームが見つかりません。"


def check_alarms_and_notify():
    """アラームをチェックして通知（バックグラウンドスレッド用）"""
    global running, alarms, client, audio

    last_triggered = {}  # 同じアラームが連続で鳴らないように

    while running:
        try:
            now = datetime.now()
            current_time = now.strftime("%H:%M")

            for alarm in alarms:
                if not alarm.get("enabled", True):
                    continue

                alarm_id = alarm['id']
                alarm_time = alarm['time']

                # 同じ分に複数回鳴らないようにチェック
                trigger_key = f"{alarm_id}_{current_time}"
                if trigger_key in last_triggered:
                    continue

                if alarm_time == current_time:
                    print(f"アラーム発動: {alarm['label']} ({alarm_time})")
                    last_triggered[trigger_key] = True

                    # 録音中でなければ通知
                    with record_lock:
                        if not is_recording:
                            try:
                                # TTSで読み上げ
                                message = alarm.get('message', f"{alarm['label']}の時間です")
                                speech_audio = text_to_speech(f"アラームです。{message}")
                                play_audio(speech_audio)
                            except Exception as e:
                                print(f"アラーム通知エラー: {e}")

            # 古いトリガー記録をクリア（1分以上前のもの）
            current_minute = now.strftime("%H:%M")
            keys_to_remove = [k for k in last_triggered if not k.endswith(current_minute)]
            for k in keys_to_remove:
                del last_triggered[k]

        except Exception as e:
            print(f"アラームチェックエラー: {e}")

        time.sleep(10)  # 10秒ごとにチェック


def start_alarm_thread():
    """アラーム監視スレッドを開始"""
    global alarm_thread
    alarm_thread = threading.Thread(target=check_alarms_and_notify, daemon=True)
    alarm_thread.start()
    print("アラーム監視スレッド開始")


# ==================== カメラ機能 ====================

def camera_capture():
    """カメラで写真を撮影"""
    try:
        # 一時ファイルパス
        image_path = "/tmp/ai_necklace_capture.jpg"

        # rpicam-stillで撮影（高速モードで撮影）
        result = subprocess.run(
            ["rpicam-still", "-o", image_path, "-t", "500", "--width", "1280", "--height", "960"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print(f"カメラエラー: {result.stderr}")
            return None, "カメラでの撮影に失敗しました"

        # 画像をbase64エンコード
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        print(f"撮影成功: {image_path}")
        return image_data, None

    except subprocess.TimeoutExpired:
        return None, "カメラの撮影がタイムアウトしました"
    except FileNotFoundError:
        return None, "rpicam-stillコマンドが見つかりません。カメラが正しく設定されていない可能性があります"
    except Exception as e:
        return None, f"カメラエラー: {str(e)}"


def camera_describe(prompt="この画像に何が写っていますか？簡潔に説明してください。"):
    """カメラで撮影してGPT-4oで画像を解析"""
    global client

    print("カメラで撮影中...")
    image_data, error = camera_capture()

    if error:
        return error

    print("画像を解析中...")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt + "\n\n日本語で回答してください。音声で読み上げるため、1-2文程度の簡潔な説明をお願いします。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "low"  # 高速化のためlowを使用
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"画像解析エラー: {str(e)}"


def init_gmail():
    """Gmail API初期化"""
    global gmail_service

    creds = None
    token_path = CONFIG["gmail_token_path"]
    credentials_path = CONFIG["gmail_credentials_path"]

    # トークンファイルが存在する場合は読み込み
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, GMAIL_SCOPES)

    # トークンが無効または存在しない場合は認証フロー
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                print(f"警告: Gmail認証情報が見つかりません: {credentials_path}")
                print("Gmail機能は無効です。")
                return False

            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)

        # トークンを保存
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    try:
        gmail_service = build('gmail', 'v1', credentials=creds)
        print("Gmail API初期化完了")
        return True
    except Exception as e:
        print(f"Gmail API初期化エラー: {e}")
        return False


def gmail_list(query="is:unread", max_results=5):
    """メール一覧を取得"""
    global gmail_service, last_email_list

    if not gmail_service:
        return "Gmail機能が初期化されていません"

    try:
        results = gmail_service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()

        messages = results.get('messages', [])

        if not messages:
            return "該当するメールはありません"

        email_list = []
        last_email_list = []

        for i, msg in enumerate(messages, 1):
            msg_detail = gmail_service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='metadata',
                metadataHeaders=['From', 'Subject', 'Date']
            ).execute()

            headers = {h['name']: h['value'] for h in msg_detail.get('payload', {}).get('headers', [])}

            # 送信者名を抽出
            from_header = headers.get('From', '不明')
            from_match = re.match(r'(.+?)\s*<', from_header)
            from_name = from_match.group(1).strip() if from_match else from_header.split('@')[0]

            email_info = {
                'id': msg['id'],
                'from': from_name,
                'from_email': from_header,  # 返信用に完全なメールアドレスを保持
                'subject': headers.get('Subject', '(件名なし)'),
                'date': headers.get('Date', ''),
            }
            last_email_list.append(email_info)
            print(f"メール保存: ID={msg['id']}, From={from_header}")  # デバッグログ
            email_list.append(f"{i}. {from_name}さんから: {email_info['subject']}")

        return "メール一覧:\n" + "\n".join(email_list)

    except HttpError as e:
        return f"メール取得エラー: {e}"


def gmail_read(message_id):
    """メール本文を読み取り"""
    global gmail_service

    if not gmail_service:
        return "Gmail機能が初期化されていません"

    try:
        msg = gmail_service.users().messages().get(
            userId='me',
            id=message_id,
            format='full'
        ).execute()

        headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}

        # 本文を取得
        body = ""
        payload = msg.get('payload', {})

        if 'body' in payload and payload['body'].get('data'):
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        elif 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain' and part.get('body', {}).get('data'):
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break

        # 長すぎる場合は切り詰め
        if len(body) > 500:
            body = body[:500] + "...(以下省略)"

        from_header = headers.get('From', '不明')
        from_match = re.match(r'(.+?)\s*<', from_header)
        from_name = from_match.group(1).strip() if from_match else from_header

        return f"送信者: {from_name}\n件名: {headers.get('Subject', '(件名なし)')}\n\n本文:\n{body}"

    except HttpError as e:
        return f"メール読み取りエラー: {e}"


def gmail_send(to, subject, body):
    """新規メール送信"""
    global gmail_service

    if not gmail_service:
        return "Gmail機能が初期化されていません"

    try:
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        gmail_service.users().messages().send(
            userId='me',
            body={'raw': raw}
        ).execute()

        return f"{to}にメールを送信しました"

    except HttpError as e:
        return f"メール送信エラー: {e}"


def gmail_send_photo(to=None, subject="写真を送ります", body="", take_photo=True):
    """写真付きメール送信（toが省略された場合は直前のメール相手に送信）"""
    global gmail_service, last_email_list

    if not gmail_service:
        return "Gmail機能が初期化されていません"

    # toが指定されていない場合、直前のメール相手を使用
    if not to:
        if not last_email_list:
            return "送信先が指定されていません。先に「メールを確認して」と言うか、宛先を指定してください。"
        # 直前のメール一覧の最初の送信者を使用
        to = extract_email_address(last_email_list[0].get('from_email', ''))
        if not to:
            return "直前のメール送信者のアドレスが取得できませんでした"
        print(f"直前のメール相手に送信: {to}")

    try:
        # 写真を撮影
        if take_photo:
            print("写真を撮影中...")
            image_path = "/tmp/ai_necklace_capture.jpg"
            result = subprocess.run(
                ["rpicam-still", "-o", image_path, "-t", "500", "--width", "1280", "--height", "960"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return f"写真の撮影に失敗しました: {result.stderr}"
        else:
            image_path = "/tmp/ai_necklace_capture.jpg"
            if not os.path.exists(image_path):
                return "送信する写真がありません。先に写真を撮影してください。"

        # MIMEマルチパートメッセージを作成
        message = MIMEMultipart()
        message['to'] = to
        message['subject'] = subject

        # 本文を追加
        if body:
            message.attach(MIMEText(body, 'plain'))
        else:
            message.attach(MIMEText("写真を送ります。", 'plain'))

        # 画像を添付
        with open(image_path, 'rb') as f:
            img_data = f.read()

        img_part = MIMEBase('image', 'jpeg')
        img_part.set_payload(img_data)
        encoders.encode_base64(img_part)

        # ファイル名を設定（日時を含める）
        filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        img_part.add_header('Content-Disposition', 'attachment', filename=filename)
        message.attach(img_part)

        # 送信
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        gmail_service.users().messages().send(
            userId='me',
            body={'raw': raw}
        ).execute()

        # 送信先の名前を抽出
        to_match = re.match(r'(.+?)\s*<', to)
        to_name = to_match.group(1).strip() if to_match else to.split('@')[0]

        return f"{to_name}さんに写真付きメールを送信しました"

    except subprocess.TimeoutExpired:
        return "カメラの撮影がタイムアウトしました"
    except FileNotFoundError:
        return "カメラが見つかりません"
    except HttpError as e:
        return f"メール送信エラー: {e}"
    except Exception as e:
        return f"写真付きメール送信エラー: {str(e)}"


def extract_email_address(email_str):
    """メールアドレス部分を抽出（例: '"名前" <test@example.com>' → 'test@example.com'）"""
    if not email_str:
        return None
    # <email@example.com> 形式からメールアドレスを抽出
    match = re.search(r'<([^>]+)>', email_str)
    if match:
        return match.group(1)
    # @が含まれていればそのまま使用
    if '@' in email_str:
        return email_str.strip()
    return None


def gmail_reply(message_id, body, to_email=None, attach_photo=False):
    """メール返信（写真添付オプション付き）"""
    global gmail_service

    if not gmail_service:
        return "Gmail機能が初期化されていません"

    try:
        # 写真添付が必要な場合は撮影
        image_path = None
        if attach_photo:
            print("写真を撮影中...")
            image_path = "/tmp/ai_necklace_capture.jpg"
            result = subprocess.run(
                ["rpicam-still", "-o", image_path, "-t", "500", "--width", "1280", "--height", "960"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return f"写真の撮影に失敗しました: {result.stderr}"

        # 元のメールを取得
        original = gmail_service.users().messages().get(
            userId='me',
            id=message_id,
            format='metadata',
            metadataHeaders=['From', 'Subject', 'Message-ID', 'References', 'Reply-To']
        ).execute()

        headers = {h['name']: h['value'] for h in original.get('payload', {}).get('headers', [])}

        # 返信先（Reply-Toがあればそれを使う、なければFrom）
        to_raw = to_email or headers.get('Reply-To') or headers.get('From', '')
        to = extract_email_address(to_raw)

        if not to:
            return "返信先のメールアドレスが取得できませんでした"

        subject = headers.get('Subject', '')
        if not subject.startswith('Re:'):
            subject = 'Re: ' + subject

        # スレッド情報
        thread_id = original.get('threadId')
        message_id_header = headers.get('Message-ID', '')
        references = headers.get('References', '')

        # 写真添付の場合はMIMEMultipart、そうでなければMIMEText
        if attach_photo and image_path:
            message = MIMEMultipart()
            message['to'] = to
            message['subject'] = subject
            if message_id_header:
                message['In-Reply-To'] = message_id_header
                message['References'] = f"{references} {message_id_header}".strip()

            # 本文を追加
            message.attach(MIMEText(body or "写真を送ります。", 'plain'))

            # 画像を添付
            with open(image_path, 'rb') as f:
                img_data = f.read()
            img_part = MIMEBase('image', 'jpeg')
            img_part.set_payload(img_data)
            encoders.encode_base64(img_part)
            filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            img_part.add_header('Content-Disposition', 'attachment', filename=filename)
            message.attach(img_part)
        else:
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            if message_id_header:
                message['In-Reply-To'] = message_id_header
                message['References'] = f"{references} {message_id_header}".strip()

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        gmail_service.users().messages().send(
            userId='me',
            body={'raw': raw, 'threadId': thread_id}
        ).execute()

        # 送信先の名前を抽出
        to_match = re.match(r'(.+?)\s*<', to)
        to_name = to_match.group(1).strip() if to_match else to.split('@')[0]

        if attach_photo:
            return f"{to_name}さんに写真付きで返信しました"
        return f"{to_name}さんに返信を送信しました"

    except subprocess.TimeoutExpired:
        return "カメラの撮影がタイムアウトしました"
    except HttpError as e:
        return f"返信エラー: {e}"


def execute_tool(tool_call):
    """ツール呼び出しを実行"""
    global last_email_list

    tool_name = tool_call.get('tool')
    params = tool_call.get('params', {})

    if tool_name == 'gmail_list':
        return gmail_list(
            query=params.get('query', 'is:unread'),
            max_results=params.get('max_results', 5)
        )
    elif tool_name == 'gmail_read':
        # 番号で指定された場合
        msg_id = params.get('message_id')
        if isinstance(msg_id, int) or (isinstance(msg_id, str) and msg_id.isdigit()):
            idx = int(msg_id) - 1
            if 0 <= idx < len(last_email_list):
                msg_id = last_email_list[idx]['id']
            else:
                return "指定されたメールが見つかりません"
        return gmail_read(msg_id)
    elif tool_name == 'gmail_send':
        return gmail_send(
            to=params.get('to'),
            subject=params.get('subject'),
            body=params.get('body')
        )
    elif tool_name == 'gmail_reply':
        msg_id = params.get('message_id')
        to_email = None
        attach_photo = params.get('attach_photo', False)
        print(f"gmail_reply: params={params}, attach_photo={attach_photo}")  # デバッグログ
        if isinstance(msg_id, int) or (isinstance(msg_id, str) and msg_id.isdigit()):
            idx = int(msg_id) - 1
            print(f"返信処理: idx={idx}, last_email_list長さ={len(last_email_list)}")  # デバッグログ
            if 0 <= idx < len(last_email_list):
                msg_id = last_email_list[idx]['id']
                to_email = last_email_list[idx].get('from_email')
                print(f"返信先: msg_id={msg_id}, to_email={to_email}")  # デバッグログ
            else:
                return "指定されたメールが見つかりません。先に「メールを確認して」と言ってください。"
        return gmail_reply(msg_id, params.get('body'), to_email, attach_photo)
    # アラーム機能
    elif tool_name == 'alarm_set':
        return alarm_set(
            time_str=params.get('time'),
            label=params.get('label', 'アラーム'),
            message=params.get('message', '')
        )
    elif tool_name == 'alarm_list':
        return alarm_list()
    elif tool_name == 'alarm_delete':
        return alarm_delete(params.get('alarm_id'))
    # カメラ機能
    elif tool_name == 'camera_capture':
        prompt = params.get('prompt', 'この画像に何が写っていますか？簡潔に説明してください。')
        return camera_describe(prompt)
    # 写真付きメール送信
    elif tool_name == 'gmail_send_photo':
        return gmail_send_photo(
            to=params.get('to'),
            subject=params.get('subject', '写真を送ります'),
            body=params.get('body', ''),
            take_photo=params.get('take_photo', True)
        )
    # 音声メッセージ録音・送信
    elif tool_name == 'voice_record_send':
        if not firebase_messenger:
            return "音声メッセージ機能が無効です"
        return "VOICE_RECORD_SEND"  # 特殊な戻り値で録音モードを示す
    else:
        return f"不明なツール: {tool_name}"


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


def record_audio_while_pressed():
    """ボタンを押している間録音（トランシーバー方式）"""
    global audio, button, is_recording

    input_device = CONFIG["input_device_index"]
    if input_device is None:
        input_device = find_audio_device(audio, "input")

    print("録音中... (ボタンを離すと停止)")

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CONFIG["channels"],
        rate=CONFIG["sample_rate"],
        input=True,
        input_device_index=input_device,
        frames_per_buffer=CONFIG["chunk_size"],
        stream_callback=None
    )

    frames = []
    max_chunks = int(CONFIG["sample_rate"] / CONFIG["chunk_size"] * CONFIG["max_record_seconds"])

    # タイムアウト設定（60秒）
    recording_timeout = 60
    start_time = time.time()

    with record_lock:
        is_recording = True

    while True:
        if not running:
            break

        # タイムアウトチェック
        elapsed_time = time.time() - start_time
        if elapsed_time > recording_timeout:
            print(f"録音タイムアウト ({recording_timeout}秒経過)、録音終了")
            break

        # ボタンチェック（最優先）
        if button and not button.is_pressed:
            print("ボタンが離されました、録音終了")
            break

        # 最大録音時間チェック
        if len(frames) >= max_chunks:
            print("最大録音時間に達しました、録音終了")
            break

        try:
            # stream.get_read_available()でデータが利用可能かチェック
            available = stream.get_read_available()
            if available >= CONFIG["chunk_size"]:
                data = stream.read(CONFIG["chunk_size"], exception_on_overflow=False)
                frames.append(data)
            else:
                # データがまだ準備できていない場合は短時間待機（ボタンチェック優先のため短く）
                time.sleep(0.001)  # 1msに短縮
        except Exception as e:
            print(f"録音中にエラー: {e}")
            break

    with record_lock:
        is_recording = False

    stream.stop_stream()
    stream.close()

    if len(frames) < 5:
        print("録音が短すぎます")
        return None

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(CONFIG["channels"])
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(CONFIG["sample_rate"])
        wf.writeframes(b''.join(frames))

    wav_buffer.seek(0)
    return wav_buffer


def record_audio_auto():
    """自動録音（ボタンなしモード、無音検出で停止）"""
    global audio

    input_device = CONFIG["input_device_index"]
    if input_device is None:
        input_device = find_audio_device(audio, "input")

    print("録音開始... 話しかけてください")

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CONFIG["channels"],
        rate=CONFIG["sample_rate"],
        input=True,
        input_device_index=input_device,
        frames_per_buffer=CONFIG["chunk_size"]
    )

    frames = []
    silent_chunks = 0
    has_sound = False
    max_chunks = int(CONFIG["sample_rate"] / CONFIG["chunk_size"] * 5)
    silence_duration = 1.5
    silence_chunks_threshold = int(CONFIG["sample_rate"] / CONFIG["chunk_size"] * silence_duration)

    for i in range(max_chunks):
        if not running:
            break

        data = stream.read(CONFIG["chunk_size"], exception_on_overflow=False)
        frames.append(data)

        audio_data = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_data).mean()

        if volume > CONFIG["silence_threshold"]:
            has_sound = True
            silent_chunks = 0
        else:
            silent_chunks += 1

        if has_sound and silent_chunks > silence_chunks_threshold:
            print("無音検出、録音終了")
            break

    stream.stop_stream()
    stream.close()

    if not has_sound:
        print("音声が検出されませんでした")
        return None

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(CONFIG["channels"])
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(CONFIG["sample_rate"])
        wf.writeframes(b''.join(frames))

    wav_buffer.seek(0)
    return wav_buffer


def transcribe_audio(audio_data):
    """音声をテキストに変換（Whisper API）"""
    global client

    print("音声認識中...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_data.read())
        temp_path = f.name

    try:
        with open(temp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=CONFIG["whisper_model"],
                file=audio_file,
                language="ja"
            )
        return transcript.text
    finally:
        os.unlink(temp_path)


def get_ai_response(text):
    """AIからの応答を取得（ツール呼び出し対応）"""
    global client, conversation_history

    print(f"AI処理中... (入力: {text})")

    # 会話履歴に追加
    conversation_history.append({"role": "user", "content": text})

    # 履歴が長くなりすぎたら古いものを削除
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]

    messages = [
        {"role": "system", "content": CONFIG["system_prompt"]}
    ] + conversation_history

    response = client.chat.completions.create(
        model=CONFIG["chat_model"],
        messages=messages,
        max_tokens=500
    )

    ai_response = response.choices[0].message.content
    print(f"GPT応答: {ai_response}")  # デバッグログ

    # ツール呼び出しかチェック（応答内にJSONが含まれているか）
    try:
        # JSON形式のツール呼び出しを検出（応答の中からJSONを抽出）
        # {"tool": "...", "params": {...}} 形式を探す
        # ネストした括弧に対応するため、より柔軟なパターンを使用
        json_match = re.search(r'\{"tool":\s*"[^"]+",\s*"params":\s*\{[^{}]*\}\}', ai_response)
        if not json_match:
            # paramsが空または単純な値の場合
            json_match = re.search(r'\{"tool":\s*"[^"]+",\s*"params":\s*\{[^}]*\}\}', ai_response)
        if not json_match:
            # シンプルな形式も試す
            json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', ai_response)

        # マッチした文字列からJSONをパース（失敗したら全体から抽出を試みる）
        tool_call = None
        if json_match:
            json_str = json_match.group()
            try:
                tool_call = json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # 正規表現でうまくいかない場合、{ から } までを順番に試す
        if not tool_call and '"tool"' in ai_response:
            start_idx = ai_response.find('{"tool"')
            if start_idx == -1:
                start_idx = ai_response.find('{ "tool"')
            if start_idx != -1:
                # 対応する閉じ括弧を探す
                depth = 0
                for i, c in enumerate(ai_response[start_idx:]):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = ai_response[start_idx:start_idx + i + 1]
                            try:
                                tool_call = json.loads(json_str)
                                break
                            except json.JSONDecodeError:
                                continue

        if tool_call and 'tool' in tool_call:
            print(f"ツール呼び出し: {tool_call}")

            # ツール実行
            tool_result = execute_tool(tool_call)
            print(f"ツール結果: {tool_result}")

            # 音声録音・送信の特殊処理
            if tool_result == "VOICE_RECORD_SEND":
                return "VOICE_RECORD_SEND"

            # ツール結果を含めて再度AIに問い合わせ
            conversation_history.append({"role": "assistant", "content": ai_response})
            conversation_history.append({"role": "user", "content": f"ツール実行結果:\n{tool_result}\n\nこの結果を音声で読み上げるために、簡潔に日本語で要約してください。"})

            messages = [
                {"role": "system", "content": CONFIG["system_prompt"]}
            ] + conversation_history

            summary_response = client.chat.completions.create(
                model=CONFIG["chat_model"],
                messages=messages,
                max_tokens=300
            )

            final_response = summary_response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": final_response})
            return final_response

    except json.JSONDecodeError:
        pass  # JSONでない場合は通常の応答として処理

    conversation_history.append({"role": "assistant", "content": ai_response})
    return ai_response


def text_to_speech(text):
    """テキストを音声に変換（TTS API）"""
    global client

    print(f"音声合成中... (テキスト: {text[:30]}...)")

    response = client.audio.speech.create(
        model=CONFIG["tts_model"],
        voice=CONFIG["tts_voice"],
        input=text,
        speed=CONFIG["tts_speed"],
        response_format="wav"
    )

    return response.content


def play_audio(audio_data):
    """音声を再生"""
    global audio

    if audio_data is None:
        print("音声データがありません")
        return

    output_device = CONFIG["output_device_index"]
    if output_device is None:
        output_device = find_audio_device(audio, "output")

    print("再生中...")

    wav_buffer = io.BytesIO(audio_data)
    with wave.open(wav_buffer, 'rb') as wf:
        original_rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        target_rate = 48000

        # 全フレームを読み込み
        frames = wf.readframes(wf.getnframes())

        # 48000Hz以外の場合はリサンプリング
        if original_rate != target_rate:
            # バイトデータをnumpy配列に変換
            audio_array = np.frombuffer(frames, dtype=np.int16)

            # リサンプリング（線形補間）
            original_length = len(audio_array)
            target_length = int(original_length * target_rate / original_rate)
            indices = np.linspace(0, original_length - 1, target_length)
            resampled = np.interp(indices, np.arange(original_length), audio_array)
            frames = resampled.astype(np.int16).tobytes()
            print(f"リサンプリング: {original_rate}Hz → {target_rate}Hz")

        stream = audio.open(
            format=audio.get_format_from_width(sampwidth),
            channels=channels,
            rate=target_rate,
            output=True,
            output_device_index=output_device
        )

        chunk_size = 1024 * sampwidth * channels
        for i in range(0, len(frames), chunk_size):
            if not running:
                break
            stream.write(frames[i:i+chunk_size])

        stream.stop_stream()
        stream.close()


def process_voice():
    """音声処理のメインフロー"""
    global button

    if CONFIG["use_button"] and button:
        audio_data = record_audio_while_pressed()
    else:
        audio_data = record_audio_auto()

    if audio_data is None:
        return

    try:
        text = transcribe_audio(audio_data)
        if not text or text.strip() == "":
            print("テキストが認識できませんでした")
            return

        print(f"\n[あなた] {text}")

        response = get_ai_response(text)
        print(f"[AI] {response}")

        # 音声メッセージ録音・送信モードの処理
        if response == "VOICE_RECORD_SEND":
            record_and_send_voice_message()
            return

        speech_audio = text_to_speech(response)
        play_audio(speech_audio)

    except APIConnectionError as e:
        print(f"⚠️ ネットワーク接続エラー: {e}")
        print("インターネット接続を確認してください。再試行できます。")
    except Exception as e:
        print(f"⚠️ 処理エラー: {e}")
        import traceback
        traceback.print_exc()


def record_and_send_voice_message():
    """音声を録音してスマホに送信"""
    global button, firebase_messenger

    # 録音開始のアナウンス
    announce = text_to_speech("了解です。押しながら話してください。")
    play_audio(announce)

    # 録音
    print("📢 メッセージを録音中...")
    if CONFIG["use_button"] and button:
        # ボタンが押されるまで待機
        print("ボタンを押して録音を開始してください...")
        while not button.is_pressed and running:
            time.sleep(0.05)
        if not running:
            return
        audio_data = record_audio_while_pressed()
    else:
        audio_data = record_audio_auto()

    if audio_data is None:
        print("録音に失敗しました")
        error_msg = text_to_speech("録音に失敗しました")
        play_audio(error_msg)
        return

    # 音声をテキストに変換（Whisper API）
    print("🔤 音声をテキストに変換中...")
    audio_data.seek(0)  # バッファの先頭に戻す
    transcribed_text = None
    try:
        transcribed_text = transcribe_audio(audio_data)
        if transcribed_text:
            print(f"変換されたテキスト: {transcribed_text}")
    except Exception as e:
        print(f"テキスト変換エラー: {e}")

    # スマホに送信（テキスト付き）
    print("📤 スマホに送信中...")
    audio_data.seek(0)  # バッファの先頭に戻す
    if send_voice_to_phone(audio_data, text=transcribed_text):
        success_msg = text_to_speech("メッセージをスマホに送信しました")
        play_audio(success_msg)
    else:
        error_msg = text_to_speech("送信に失敗しました")
        play_audio(error_msg)


def main():
    """メインループ"""
    global running, client, audio, button

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("エラー: OPENAI_API_KEY が設定されていません")
        print(".env ファイルに OPENAI_API_KEY=sk-... を設定してください")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    audio = pyaudio.PyAudio()

    # Gmail初期化
    gmail_available = init_gmail()

    # アラーム初期化
    load_alarms()
    start_alarm_thread()

    # Firebase Voice Messenger初期化
    firebase_available = init_firebase_messenger()

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

    print("=" * 50)
    print("AI Necklace 起動 (Gmail・アラーム・音声メッセージ機能付き)")
    print("=" * 50)
    print(f"Chat Model: {CONFIG['chat_model']}")
    print(f"TTS Voice: {CONFIG['tts_voice']}")
    print(f"Gmail: {'有効' if gmail_available else '無効'}")
    print(f"Voice Messenger: {'有効' if firebase_available else '無効'}")
    if CONFIG["use_button"]:
        print(f"操作方法: GPIO{CONFIG['button_pin']}のボタンを押している間録音")
    else:
        print("操作方法: 自動録音（無音検出で停止）")
    print("Ctrl+C で終了")
    print("=" * 50)

    if gmail_available:
        print("\nGmailコマンド例:")
        print("  - 「メールを確認して」")
        print("  - 「未読メールを読んで」")
        print("  - 「1番目のメールを読んで」")
        print("  - 「○○にメールを送って」")

    print("\nアラームコマンド例:")
    print("  - 「7時にアラームをセットして」")
    print("  - 「アラームを確認して」")
    print("  - 「アラームを削除して」")
    print(f"  現在のアラーム: {len(alarms)}件")

    print("\nカメラコマンド例:")
    print("  - 「写真を撮って」「何が見える？」")
    print("  - 「さっきの人に写真を送って」")
    print("  - 「このメールに写真付きで返信して」")
    print("=" * 50)

    try:
        if CONFIG["use_button"] and button:
            print("\n--- ボタンを押して話しかけてください ---")
            while running:
                try:
                    if button.is_pressed:
                        process_voice()
                        if running:
                            print("\n--- ボタンを押して話しかけてください ---")
                    time.sleep(0.05)
                except Exception as e:
                    print(f"⚠️ ループ内エラー: {e}")
                    import traceback
                    traceback.print_exc()
                    print("処理を継続します...")
                    time.sleep(1)
        else:
            while running:
                try:
                    print("\n--- 待機中 (話しかけてください) ---")
                    process_voice()
                except Exception as e:
                    print(f"⚠️ ループ内エラー: {e}")
                    import traceback
                    traceback.print_exc()
                    print("処理を継続します...")
                    time.sleep(1)

    except KeyboardInterrupt:
        print("\n終了シグナルを受信しました")
    except Exception as e:
        print(f"致命的エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if audio:
            audio.terminate()
        print("終了しました")


if __name__ == "__main__":
    main()

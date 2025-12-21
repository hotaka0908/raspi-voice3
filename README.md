# AI Necklace Realtime - Raspberry Pi 5 リアルタイム音声AIアシスタント

OpenAI Realtime APIを使用したリアルタイム双方向音声対話システム。

## 概要

このプロジェクトは [raspi-voice](https://github.com/hotaka0908/raspi-voice) をベースに、
OpenAI Realtime API を使用してリアルタイム音声対話を実現します。

## 従来版との比較

| 項目 | 従来版 (raspi-voice) | リアルタイム版 (本プロジェクト) |
|------|---------------------|-------------------------------|
| 方式 | 録音→STT→LLM→TTS→再生 | リアルタイムストリーミング |
| レイテンシ | 3-5秒 | 0.5-1秒 |
| 会話 | ボタン操作 | 割り込み可能 |
| API | Whisper + GPT + TTS | Realtime API |

## 機能

- **リアルタイム音声対話** - 低レイテンシの双方向会話
- **Gmail連携** - メール確認・返信・送信
- **アラーム機能** - 時刻指定で音声通知
- **カメラ機能** - GPT-4o Visionで画像認識
- **写真付きメール送信**
- **音声メッセージ** - Firebase経由でスマホとボイスメッセージをやり取り
- **写真送信** - 撮影した写真をスマホに送信

## 使用モデル

- `gpt-4o-realtime-preview-2024-12-17` - リアルタイム音声対話
- `gpt-4o` - 画像認識（カメラ機能）
- `whisper-1` - 音声メッセージの文字起こし

## ハードウェア

- Raspberry Pi 5
- USBマイク
- USBスピーカー
- プッシュボタン（GPIO5接続）
- USBカメラ（オプション）

## セットアップ

### 1. 依存関係のインストール

```bash
sudo apt-get install -y portaudio19-dev python3-pyaudio python3-lgpio python3-gpiozero ffmpeg

cd ~
mkdir ai-necklace-realtime && cd ai-necklace-realtime
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install openai python-dotenv websockets numpy
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
pip install firebase-admin
```

### 2. 環境変数の設定

```bash
mkdir -p ~/.ai-necklace
cat > ~/.ai-necklace/.env << EOF
OPENAI_API_KEY=sk-your-api-key
EOF
```

### 3. Gmail設定（オプション）

1. [Google Cloud Console](https://console.cloud.google.com/)でプロジェクトを作成
2. Gmail APIを有効化
3. OAuth 2.0クライアントIDを作成し、`credentials.json`をダウンロード
4. `~/.ai-necklace/credentials.json`に配置
5. 初回実行時にブラウザで認証（`token.json`が自動生成される）

### 4. Firebase設定（音声メッセージ用、オプション）

1. [Firebase Console](https://console.firebase.google.com/)でプロジェクトを作成
2. Realtime DatabaseとStorageを有効化
3. `firebase_voice_config.py`を作成:

```python
FIREBASE_CONFIG = {
    "apiKey": "your-api-key",
    "databaseURL": "https://your-project.firebaseio.com",
    "storageBucket": "your-project.appspot.com"
}
```

## 使い方

### 基本操作

1. ボタンを押す → 録音開始（AIが話していたら割り込み）
2. 話す（ボタンを押している間）
3. ボタンを離す → 録音終了 → AI応答開始（リアルタイムで再生）

### コマンド例

- 「メールを確認して」
- 「写真を撮って」「何が見える？」
- 「7時にアラームをセット」
- 「スマホにメッセージを送って」
- 「スマホに写真を送って」

### 手動実行

```bash
cd ~/ai-necklace-realtime
source venv/bin/activate
python ai_necklace_realtime.py
```

### サービスとして実行

```bash
sudo cp ai-necklace.service /etc/systemd/system/ai-necklace-realtime.service
sudo systemctl daemon-reload
sudo systemctl enable ai-necklace-realtime
sudo systemctl start ai-necklace-realtime
```

### ログの確認

```bash
sudo journalctl -u ai-necklace-realtime -f
```

## 実装状況

- [x] 基本的なWebSocket接続
- [x] 音声入力ストリーミング
- [x] 音声出力ストリーミング
- [x] 割り込み処理（ボタン押下でAI応答を中断）
- [x] 24kHz → 48kHz リサンプリング（USBスピーカー対応）
- [x] Gmail機能の統合
- [x] アラーム機能の統合
- [x] カメラ機能の統合
- [x] 音声メッセージ機能（Firebase経由）
- [x] 写真送信機能（Firebase経由でスマホに送信）
- [ ] VAD（Voice Activity Detection）自動モード

## ライセンス

MIT License

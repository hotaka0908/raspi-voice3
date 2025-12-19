# AI Necklace Realtime - Raspberry Pi 5 リアルタイム音声AIアシスタント

OpenAI Realtime APIを使用したリアルタイム双方向音声対話システム。

## 概要

このプロジェクトは [raspi-voice](https://github.com/hotaka0908/raspi-voice) をベースに、
OpenAI Realtime API (`gpt-realtime-mini-2025-12-15`) を使用してリアルタイム音声対話を実現します。

## 従来版との比較

| 項目 | 従来版 (raspi-voice) | リアルタイム版 (本プロジェクト) |
|------|---------------------|-------------------------------|
| 方式 | 録音→STT→LLM→TTS→再生 | リアルタイムストリーミング |
| レイテンシ | 3-5秒 | 0.5-1秒 |
| 会話 | ボタン操作 | 割り込み可能 |
| API | Whisper + GPT + TTS | Realtime API |

## 使用モデル

- `gpt-4o-realtime-preview-2024-12-17` - リアルタイム音声対話

## ハードウェア

- Raspberry Pi 5
- USBマイク
- USBスピーカー
- プッシュボタン（GPIO5接続、オプション）

## セットアップ

### 1. 依存関係のインストール

```bash
sudo apt-get install -y portaudio19-dev python3-pyaudio python3-lgpio python3-gpiozero

cd ~
mkdir ai-necklace-realtime && cd ai-necklace-realtime
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install openai python-dotenv websockets numpy
```

### 2. 環境変数の設定

```bash
mkdir -p ~/.ai-necklace
echo "OPENAI_API_KEY=sk-your-api-key" > ~/.ai-necklace/.env
```

## 使い方

### 基本操作

1. ボタンを押す → 録音開始（AIが話していたら割り込み）
2. 話す（ボタンを押している間）
3. ボタンを離す → 録音終了 → AI応答開始（リアルタイムで再生）

### 手動実行

```bash
cd ~/ai-necklace-realtime
source venv/bin/activate
python ai_necklace_realtime.py
```

### サービスとして実行

```bash
sudo cp ai-necklace.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-necklace
sudo systemctl start ai-necklace
```

## 実装状況

- [x] 基本的なWebSocket接続
- [x] 音声入力ストリーミング
- [x] 音声出力ストリーミング
- [x] 割り込み処理（ボタン押下でAI応答を中断）
- [x] 24kHz → 48kHz リサンプリング（USBスピーカー対応）
- [ ] VAD（Voice Activity Detection）自動モード
- [ ] Gmail機能の統合
- [ ] アラーム機能の統合
- [ ] カメラ機能の統合

## ライセンス

MIT License

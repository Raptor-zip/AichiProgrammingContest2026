# 📚 プリント仕分けシステム

> **AichiProgrammingContest2026**
>
> 紙のプリントをスキャンし、科目ごとに自動で仕分けるシステム

中高生の学習支援を目的とした、ArUcoマーカーを使った画像認識ベースのプリント管理ツールです。
紙にArUcoマーカーのシールを貼ることで、カメラで撮影した際に科目を自動判別し、適切なフォルダに保存します。

## 特徴

- **自動科目判別**: ArUcoマーカーを読み取って、プリントの科目を自動判別
- **自動補正機能**:
  - ホワイトバランス自動調整（6×6グリッド解析）
  - 回転補正（マーカーの傾きを検出して自動補正）
- **OCR機能**: 撮影した画像からテキストを抽出（日本語・英語対応）
- **AI学習支援機能**:
-   📝 要約: テキストの重要ポイントを箇条書きで整理
    ❓ 練習問題作成: 内容に基づいた学習問題を3問生成
    💡 概念説明: 専門用語を分かりやすく解説
    📅 学習計画: 効果的な学習スケジュールを提案

## インストール

1. **リポジトリのクローン**
```bash
git clone https://github.com/Raptor-zip/AichiProgrammingContest2026.git
cd AichiProgrammingContest2026
```

2. **依存パッケージのインストール**
```bash
pip install -r requirements.txt
```

3.AI学習支援環境のセットアップ
Ollamaのインストール:
```bash
curl -fsSL https://ollama.com/install.sh | sh￥
```
AIモデルのダウンロード:
```bash
# 軽量な日本語対応モデル（約1.6GB）
ollama pull gemma2:2b

# より高性能なモデル（約4.1GB、推奨）
ollama pull gemma2:9b
```

## 起動方法

**通常モード:**
```bash
python main.py
```

**デバッグモード:**（デバッグ用画像が保存されます）
```bash
python main.py --debug
# または
python main.py -d
```

## 技術スタック

- **GUI**: PySide6 (Qt for Python)
- **画像処理**: OpenCV
- **ArUcoマーカー検出**: cv2.aruco
- **OCR**: YomiToku
- **画像変換**: Pillow (PIL)
- **言語**: Python 3.10+
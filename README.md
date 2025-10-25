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


## 使い方

### 必要なもの

- Python 3.8以上
- Webカメラ
- ArUcoマーカー（印刷したシール）
- Tesseract OCRエンジン

### インストール

1. **リポジトリのクローン**
```bash
git clone https://github.com/Raptor-zip/AichiProgrammingContest2026.git
cd AichiProgrammingContest2026
```

2. **依存パッケージのインストール**
```bash
pip install -r requirements.txt
```

3. **Tesseract OCRのインストール**

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-jpn
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
[Tesseract公式ページ](https://github.com/UB-Mannheim/tesseract/wiki)からインストーラーをダウンロード

4.AI学習支援環境のセットアップ
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

ローカルAI用Pythonライブラリ:
```bash
pip install requests
```

### 起動方法

**通常モード:**
```bash
python main.py
```

**デバッグモード:**（グリッド解析画像も保存されます）
```bash
python main.py --debug
# または
python main.py -d
```

---

## 使用方法

### 1. 初期設定

1. アプリケーションを起動
2. 「教科設定」ボタンをクリック
3. ArUcoマーカーのIDと科目名を対応付ける
   - 例: ID `1` → `数学`、ID `2` → `英語`
4. 「保存して閉じる」をクリック

### 2. プリントの撮影

1. ArUcoマーカーのシールをプリントに貼る
2. カメラの前にプリントをかざす
3. 画面上でマーカーが認識されると、緑色の枠が表示される
4. 「撮影」ボタンをクリック（またはスペースキー）
5. 自動的に科目別フォルダに画像が保存される

### 3. OCR実行

1. 「OCR実行」ボタンをクリック
2. 最後に撮影した画像からテキストが抽出される
3. 抽出されたテキストが画面右側に表示される

### 4. ホワイトバランス調整

- 「WB: ON」ボタンで自動ホワイトバランスのON/OFF切り替え
- マーカーの白黒領域を解析して、画像の色味を自動補正

---

## ディレクトリ構造

```
AichiProgrammingContest2026/
├── main.py                    # メインアプリケーション
├── ui_components.py           # UIコンポーネント（トースト、ダイアログ）
├── ocr_worker.py             # OCR処理ワーカー（非同期処理）
├── image_processing.py       # 画像処理関数（補正、回転など）
├── subject_mappings.json     # 科目設定ファイル
├── requirements.txt          # 依存パッケージリスト
├── icon.png                  # アプリケーションアイコン
├── README.md                 # このファイル
└── captures/                 # 撮影した画像の保存先
    ├── 数学/
    ├── 英語/
    ├── 理科/
    └── 未分類/               # マーカーが認識できなかった画像
```

---

## 技術スタック

- **GUI**: PySide6 (Qt for Python)
- **画像処理**: OpenCV
- **ArUcoマーカー検出**: cv2.aruco
- **OCR**: pytesseract + Tesseract OCR
- **画像変換**: Pillow (PIL)
- **言語**: Python 3.8+

---

## 機能詳細

### ホワイトバランス補正

ArUcoマーカーを6×6のグリッドに分割し、各セルの明度を解析。白色領域と黒色領域のBGR値の中央値を取得し、画像全体の色バランスを自動調整します。

### 回転補正

マーカーの4つの角の座標から回転角度を計算し、画像を正しい向きに自動補正します。

### OCR処理

撮影した画像をバックグラウンドでOCR処理し、日本語と英語のテキストを抽出。UIスレッドをブロックしない非同期処理を実現しています。

---

## トラブルシューティング

### カメラが起動しない
- カメラが他のアプリケーションで使用されていないか確認
- カメラのアクセス許可を確認

### OCRが日本語を認識しない
- Tesseractの日本語データがインストールされているか確認
- `tesseract --list-langs`で利用可能な言語を確認

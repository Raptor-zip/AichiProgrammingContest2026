# 📚 Smart Study Assistant (Web GUI Ver.)

> **AichiProgrammingContest2026**
>
> 物理的なプリントをデジタル化・解析し、AIが学習をサポートする次世代スキャナーシステム

中高生の学習支援を目的とした、**ArUcoマーカー**を活用した自動プリント管理・学習支援ツールです。
従来のデスクトップアプリから、**モダンなWebベースのGUI**へと生まれ変わりました。PCだけでなく、タブレットやスマホのブラウザからもアクセス可能です。

---

## ✨ 主な機能

### 1. 📷 スマート・オートキャプチャ
- **ArUcoマーカー自動検知**: マーカーを認識すると、自動でカウントダウンを開始し撮影します。
- **スタイリッシュなHUD**: 撮影状況を直感的に伝える円形プログレスバーとアニメーション。
- **自動画像補正**:
  - **ホワイトバランス**: 色味を自動で自然な白さに補正。
  - **台形補正 (Perspective Transform)**: 斜めから撮影しても、真正面から見たように補正。
  - **回転補正**: マーカーの向きに合わせて画像を正位置に回転。

### 2. 🖼️ 高機能ビューア & 履歴管理
- **撮影履歴**: 撮影した画像を時系列・科目ごとに一覧表示。
- **オリジナル/加工後 切り替え**: トグルスイッチ一つで、補正前の「生データ」と補正後の画像を瞬時に比較可能。
- **科目自動振り分け**: マーカーIDに基づいて、画像を科目別のフォルダに自動で振り分けます。

### 3. 🧠 AI 学習支援 (Powered by Gemini 2.0 Flash)
- **AI解説**: プリント内の分からない箇所を選択して「解説」ボタンを押すだけで、AIが分かりやすく説明。
- **練習問題生成**: テキストの内容に基づいたオリジナルの練習問題をAIが作成。
- **Markdown表示**: AIの回答はMarkdown形式で整形され、太字やリストで見やすく表示されます。

### 4. 🔍 OCR & テキスト解析
- **YomiToku OCR**: 手書き文字や数式にも強い高性能OCRエンジンを搭載。
- **テキスト抽出**: 画像上の領域をクリックするだけで、テキストデータを抽出・コピー可能。

---

## 🛠️ 技術スタック

### Frontend
- **React 19**: 最新のReactで構築された高速なUI
- **Vite**: 超高速なビルドツール
- **Tailwind CSS**: 美しくモダンなデザイン
- **Lucide React**: シンプルで洗練されたアイコン

### Backend
- **FastAPI**: 高性能な非同期Python Webフレームワーク
- **OpenCV**: 画像処理、ArUcoマーカー検出、幾何学変換
- **Google GenAI SDK**: Google Gemini 2.0 Flash モデルによる高度な言語処理
- **YomiToku**: 高精度な日本語OCR

---

## 🚀 インストール & セットアップ

### 前提条件
- Python 3.10+
- Node.js 18+
- Google API Key (Vertex AI / Gemini API)

### 1. リポジトリのクローン
```bash
git clone https://github.com/Raptor-zip/AichiProgrammingContest2026.git
cd AichiProgrammingContest2026
```

### 2. バックエンド環境の構築
```bash
# 仮想環境の作成と有効化 (推奨)
python -m venv venv
source venv/bin/activate

# 依存パッケージのインストール
pip install -r backend/requirements.txt
```

### 3. フロントエンド環境の構築
```bash
cd frontend
npm install
cd ..
```

### 4. 環境変数の設定
プロジェクトルートに `.env` ファイルを作成し、Gemini APIキーを設定してください。
```env
GEMINI_API_KEY=AIzaSy...
```

---

## ▶️ 起動方法

以下のスクリプトを実行するだけで、バックエンドとフロントエンドが同時に起動します。

```bash
./start_app.sh
```

起動後、ブラウザで **http://localhost:5173** にアクセスしてください。

---

## 📂 ディレクトリ構成

```
.
├── backend/             # FastAPI バックエンド
│   ├── api.py           # APIエンドポイント定義
│   ├── camera_manager.py# カメラ制御 & ArUco検出
│   ├── image_processing.py # 画像処理ロジック (補正, WB)
│   └── llm_service.py   # Gemini API 連携サービス
├── frontend/            # React フロントエンド
│   ├── src/
│   │   ├── components/  # Reactコンポーネント (HUD, HistoryViewなど)
│   │   └── App.tsx      # メインアプリケーション
├── captures/            # 撮影画像の保存先 (科目ごとに自動分類)
├── config.yaml          # アプリケーション設定
└── start_app.sh         # 一括起動スクリプト
```

---

## 🎮 使い方

1. **プリントの準備**: ArUcoマーカーシールを貼ったプリントを用意します。
2. **撮影**: カメラにプリントを映します。マーカーが安定して認識されると、HUDが回転し自動で撮影されます。
3. **確認**: 画面左側の履歴から撮影した画像を選択します。
4. **学習**:
   - 右上のトグルで補正の有無を確認。
   - 画像内の文字をクリックしてテキスト化。
   - 「Explanation」や「Practice」ボタンでAIのアシストを受けます。
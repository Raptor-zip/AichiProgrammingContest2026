# Environment Setup Memo / 環境構築メモ

## 1. Prerequisites / 前提条件

### System packages (sudo required)
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
```

### Node.js (via nvm - no sudo required)
```bash
# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

# Reload shell or run:
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Install Node.js 20
nvm install 20
nvm use 20
```

## 2. Backend Setup / バックエンド構築

```bash
cd /path/to/AichiProgrammingContest2025-AppCategory

# Create virtual environment
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
```

## 3. Frontend Setup / フロントエンド構築

```bash
cd frontend
npm install
cd ..
```

## 4. Environment Variables / 環境変数

Create `.env` file in project root:
```bash
echo "GEMINI_API_KEY=YOUR_API_KEY_HERE" > .env
```

Replace `YOUR_API_KEY_HERE` with your actual Gemini API key.

## 5. Running the App / アプリ起動

### Option A: Use start script
```bash
./start_app.sh
```

### Option B: Manual start

Terminal 1 (Backend):
```bash
source venv/bin/activate
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Terminal 2 (Frontend):
```bash
# Load nvm if needed
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

cd frontend
npm run dev
```

## 6. Access / アクセス

- Frontend: http://localhost:5173
- Backend API Docs: http://localhost:8000/docs

## Version Info / バージョン情報

- Python: 3.10+ (tested with 3.12.3)
- Node.js: 18+ (tested with 20.20.0)
- npm: 10.8.2+

## Troubleshooting / トラブルシューティング

### "externally-managed-environment" error
Python 3.12+ on Ubuntu/Debian requires virtual environment. Make sure to:
1. Install python3-venv: `sudo apt install python3-venv`
2. Always use venv: `source venv/bin/activate`

### nvm command not found
After installing nvm, reload your shell:
```bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
```

### Port already in use
```bash
fuser -k 8000/tcp  # Kill backend port
fuser -k 5173/tcp  # Kill frontend port
```

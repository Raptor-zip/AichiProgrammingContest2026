import json
import os
from datetime import datetime

import requests


class LocalLLMClient:
    def __init__(self, model):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = model
        self.conversation_history = []

        # Ollamaが動作しているかチェック
        if not self.check_ollama_connection():
            print("警告: Ollamaが起動していません。")
            self.ollama_available = False
        else:
            self.ollama_available = True

    def check_ollama_connection(self):
        """Ollamaサーバーが動作しているかチェック"""
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def summarize_text(self, text):
        """テキストを要約する"""
        prompt = f"""
以下のテキストを簡潔に要約してください。主要なポイントを3-5個の箇条書きで示してください。

テキスト:
{text}

要約:
"""
        return self._send_message(prompt, system_message="あなたは教育コンテンツの要約専門家です。")

    def create_practice_problems(self, text):
        """テキストから練習問題を作成する"""
        prompt = f"""
以下のテキスト内容に基づいて、学習者向けの練習問題を3問作成してください。
各問題には解答と簡単な解説も含めてください。

テキスト:
{text}

練習問題:
"""
        return self._send_message(
            prompt, system_message="あなたは教育問題作成の専門家です。適切な難易度で学習効果の高い問題を作成します。"
        )

    def explain_concepts(self, text):
        """概念を分かりやすく説明する"""
        prompt = f"""
以下のテキストに含まれる重要な概念を、中学生・高校生にも分かりやすく説明してください。
専門用語がある場合は、簡単な言葉で言い換えて説明も加えてください。

テキスト:
{text}

説明:
"""
        return self._send_message(
            prompt, system_message="あなたは教育専門家です。複雑な概念を分かりやすく説明することが得意です。"
        )

    def create_study_plan(self, text, subject=""):
        """学習計画を作成する"""
        prompt = f"""
以下の{subject}の学習内容について、効果的な学習計画を提案してください。
学習の順序、重要ポイント、復習のタイミングなども含めて提案してください。

学習内容:
{text}

学習計画:
"""
        return self._send_message(
            prompt, system_message="あなたは学習計画の専門家です。効果的で実践的な学習方法を提案します。"
        )

    def _send_message(self, message, system_message=""):
        """ローカルLLMにメッセージを送信"""
        if not self.ollama_available:
            return "Ollamaが利用できません。Ollamaをインストールして起動してください。"

        try:
            # システムメッセージを含むプロンプトを構築
            full_prompt = ""
            if system_message:
                full_prompt += f"システム: {system_message}\n\n"
            full_prompt += f"ユーザー: {message}"

            # Ollama APIにリクエスト
            data = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 1500},
            }

            response = requests.post(self.ollama_url, json=data, timeout=120)

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "レスポンスが空です")
            else:
                return f"エラー: サーバーからの応答コード {response.status_code}"

        except requests.exceptions.Timeout:
            return "タイムアウトエラー: レスポンスに時間がかかっています。"
        except requests.exceptions.ConnectionError:
            return "接続エラー: Ollamaサーバーに接続できません。"
        except Exception as e:
            return f"エラーが発生しました: {e}"


class AIProcessingDialog:
    """AI処理の選択ダイアログ"""

    def __init__(self, parent, ocr_text, subject_name=""):
        from PySide6 import QtCore, QtWidgets

        self.parent = parent
        self.ocr_text = ocr_text
        self.subject_name = subject_name
        self.llm_client = LocalLLMClient(model="gemma2:2b")

        # ダイアログを作成
        self.dialog = QtWidgets.QDialog(parent)
        self.dialog.setWindowTitle("AI処理メニュー")
        self.dialog.resize(800, 600)

        # レイアウト
        layout = QtWidgets.QVBoxLayout(self.dialog)

        # OCRテキスト表示
        layout.addWidget(QtWidgets.QLabel("認識されたテキスト:"))
        self.text_display = QtWidgets.QTextEdit()
        self.text_display.setPlainText(ocr_text)
        self.text_display.setMaximumHeight(150)
        layout.addWidget(self.text_display)

        # 処理選択ボタン
        button_layout = QtWidgets.QHBoxLayout()

        self.summary_btn = QtWidgets.QPushButton("📝 要約")
        self.summary_btn.clicked.connect(self.summarize)
        button_layout.addWidget(self.summary_btn)

        self.problems_btn = QtWidgets.QPushButton("❓ 練習問題作成")
        self.problems_btn.clicked.connect(self.create_problems)
        button_layout.addWidget(self.problems_btn)

        self.explain_btn = QtWidgets.QPushButton("💡 概念説明")
        self.explain_btn.clicked.connect(self.explain)
        button_layout.addWidget(self.explain_btn)

        self.plan_btn = QtWidgets.QPushButton("📅 学習計画")
        self.plan_btn.clicked.connect(self.create_plan)
        button_layout.addWidget(self.plan_btn)

        layout.addLayout(button_layout)

        # 結果表示エリア
        layout.addWidget(QtWidgets.QLabel("AI処理結果:"))
        self.result_display = QtWidgets.QTextEdit()
        self.result_display.setPlaceholderText("処理結果がここに表示されます...")
        layout.addWidget(self.result_display)

        # 進行状況表示
        self.progress_label = QtWidgets.QLabel("")
        layout.addWidget(self.progress_label)

        # 閉じるボタン
        close_btn = QtWidgets.QPushButton("閉じる")
        close_btn.clicked.connect(self.dialog.close)
        layout.addWidget(close_btn)

    def show_processing(self, message):
        """処理中メッセージを表示"""
        self.progress_label.setText(f"⏳ {message}...")
        self.result_display.clear()

    def show_result(self, result):
        """結果を表示"""
        self.progress_label.setText("✅ 処理完了")
        self.result_display.setPlainText(result)

        # 結果をファイルに保存
        self.save_result(result)

    def save_result(self, result):
        """結果をファイルに保存"""
        try:
            # 保存フォルダを作成
            ai_results_dir = os.path.join(self.parent.subject_dir, "ai_results")
            print(f"AI結果保存先: {ai_results_dir}")
            print(f"self.parent.subject_dir: {self.parent.subject_dir}")
            os.makedirs(ai_results_dir, exist_ok=True)

            # ファイル名を生成
            # TODO main.pyのtsを使いたい
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(ai_results_dir, f"capture_{timestamp}_llm.txt")

            # 保存
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"教科: {self.subject_name}\n")
                f.write(f"処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"元テキスト:\n{self.ocr_text}\n\n")
                f.write(f"AI処理結果:\n{result}")

        except Exception as e:
            print(f"結果保存エラー: {e}")

    def summarize(self):
        self.show_processing("テキストを要約中")
        result = self.llm_client.summarize_text(self.ocr_text)
        self.show_result(result)

    def create_problems(self):
        self.show_processing("練習問題を作成中")
        result = self.llm_client.create_practice_problems(self.ocr_text)
        self.show_result(result)

    def explain(self):
        self.show_processing("概念を説明中")
        result = self.llm_client.explain_concepts(self.ocr_text)
        self.show_result(result)

    def create_plan(self):
        self.show_processing("学習計画を作成中")
        result = self.llm_client.create_study_plan(self.ocr_text, self.subject_name)
        self.show_result(result)

    def exec(self):
        return self.dialog.exec()


# テスト用のメイン関数
if __name__ == "__main__":
    import sys

    from PySide6 import QtWidgets

    # 簡単なテスト用のダミーテキスト
    test_text = """
    物理学とは、自然界の現象を数学的に記述し、理解しようとする学問です。
    力学、電磁気学、熱力学、量子力学などの分野があります。
    ニュートンの運動法則やマクスウェル方程式などの基本法則があります。
    """

    app = QtWidgets.QApplication(sys.argv)

    # ダミーの親ウィンドウを作成
    parent = QtWidgets.QWidget()

    # AI処理ダイアログをテスト
    dialog = AIProcessingDialog(parent, test_text, "物理")
    result = dialog.exec()

    sys.exit(result)


from google import genai
import os
from config_loader import get_config

class LLMService:
    def __init__(self):
        self.config = get_config()
        self.api_key = self.config.get_gemini_api_key()
        self.mock_mode = not self.api_key

        self.client = None
        if not self.mock_mode:
            try:
                # Initialize the new Client
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize Gemini Client: {e}")
                self.mock_mode = True

    def explain_text(self, text: str, context: str = None) -> str:
        """Generate explanation for the given text"""
        if self.mock_mode:
            return f"[MOCK] Explanation for: {text}\n\nThis is a placeholder explanation because no API key was configured."

        context_section = ""
        if context:
            context_section = f"""
【ドキュメント全体】
{context}

"""

        prompt = f"""あなたは親切な家庭教師です。学生がドキュメント内の特定の部分について質問しています。
{context_section}【質問対象のテキスト】
{text}

上記の「質問対象のテキスト」について、ドキュメント全体の文脈を踏まえて、学生向けにわかりやすく簡潔に解説してください。
専門用語が含まれる場合は、それらの定義も説明してください。テキストはOCRによって生成されたものであり、正確性は保証されません。誤字脱字は無視して、指摘しないでください。2行連続した改行は避けてください。
**必ず日本語で回答してください。感情や人格を排し、事実のみを事務的に記述してください。余計な会話文は含めないでください。**"""
        return self._generate(prompt)

    def create_problems(self, text: str, context: str = None) -> str:
        """Create practice problems based on the text"""
        if self.mock_mode:
            return f"[MOCK] Practice Problems for: {text}\n\n1. Question 1?\n2. Question 2?"

        context_section = ""
        if context:
            context_section = f"""
【ドキュメント全体】
{context}

"""

        prompt = f"""あなたは先生です。学生がドキュメント内の特定の部分について練習問題を求めています。
{context_section}【対象のテキスト】
{text}

上記の「対象のテキスト」に関連する練習問題を、ドキュメント全体の文脈を踏まえて3問作成してください（選択式または記述式）。
最後に解答を含めてください。テキストはOCRによって生成されたものであり、正確性は保証されません。誤字脱字は無視して、指摘しないでください。2行連続した改行は避けてください。
**必ず日本語で回答してください。感情や人格を排し、事実のみを事務的に記述してください。余計な会話文は含めないでください。**"""
        return self._generate(prompt)

    def chat(self, message: str, history: list, context: str = None) -> str:
        """Continue a conversation with chat history"""
        if self.mock_mode:
            return f"[MOCK] Response to: {message}"

        # Build conversation history
        history_text = ""
        for msg in history:
            role = "ユーザー" if msg["role"] == "user" else "AI"
            history_text += f"{role}: {msg['content']}\n\n"

        context_section = ""
        if context:
            context_section = f"""【参考：ドキュメント内容】
{context}

"""

        prompt = f"""あなたは親切な家庭教師です。学生と会話をしています。
{context_section}【これまでの会話】
{history_text}
ユーザー: {message}

上記の会話の流れを踏まえて、学生の質問に答えてください。
**必ず日本語で回答してください。簡潔に、わかりやすく答えてください。**"""
        return self._generate(prompt)

    def _generate(self, prompt: str, max_retries: int = 3) -> str:
        import time

        for attempt in range(max_retries):
            try:
                print(f"DEBUG: Generating content with model gemini-2.0-flash... (attempt {attempt + 1})")
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                print(f"DEBUG: Generation successful. Length: {len(response.text) if response.text else 0}")
                if not response.text:
                    print(f"DEBUG: Response text is empty/None! Candidates: {response.candidates}")
                return response.text

            except Exception as e:
                error_str = str(e)
                print(f"LLM Generation Error (attempt {attempt + 1}): {e}")

                # Check for rate limit error
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                        print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "APIの利用制限に達しました。しばらく待ってから再度お試しください。"

                # Other errors
                import traceback
                traceback.print_exc()
                return f"エラーが発生しました: {str(e)}"

        return "リクエストに失敗しました。しばらく待ってから再度お試しください。"

# Global instance
llm_service = LLMService()

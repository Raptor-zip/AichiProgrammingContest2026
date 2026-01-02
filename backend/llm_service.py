
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

    def explain_text(self, text: str) -> str:
        """Generate explanation for the given text"""
        if self.mock_mode:
            return f"[MOCK] Explanation for: {text}\n\nThis is a placeholder explanation because no API key was configured."

        prompt = f"""
あなたは親切な家庭教師です。以下のテキストを学生向けにわかりやすく、簡潔に解説してください。
専門用語が含まれる場合は、それらの定義も説明してください。テキストはOCRによって生成されたものであり、正確性は保証されません。誤字脱字は無視して、指摘しないでください。2行連続した改行は避けてください。
**必ず日本語で回答してください。感情や人格を排し、事実のみを事務的に記述してください。余計な会話文は含めないでください。**

テキスト:
{text}
"""
        return self._generate(prompt)

    def create_problems(self, text: str) -> str:
        """Create practice problems based on the text"""
        if self.mock_mode:
            return f"[MOCK] Practice Problems for: {text}\n\n1. Question 1?\n2. Question 2?"

        prompt = f"""
あなたは先生です。以下のテキストの内容に基づいて、練習問題を3問作成してください（選択式または記述式）。
最後に解答を含めてください。テキストはOCRによって生成されたものであり、正確性は保証されません。誤字脱字は無視して、指摘しないでください。2行連続した改行は避けてください。
**必ず日本語で回答してください。感情や人格を排し、事実のみを事務的に記述してください。余計な会話文は含めないでください。**

テキスト:
{text}
"""
        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        try:
            print(f"DEBUG: Generating content with model gemini-2.0-flash...")
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            print(f"DEBUG: Generation successful. Length: {len(response.text) if response.text else 0}")
            if not response.text:
                 print(f"DEBUG: Response text is empty/None! Candidates: {response.candidates}")
            return response.text
        except Exception as e:
            # Fallback or error logging
            print(f"LLM Generation Error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating content: {str(e)}"

# Global instance
llm_service = LLMService()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(__file__))

from chatgpt import LocalLLMClient

def test_local_llm():
    """ローカルLLMの簡単なテスト"""
    print("=== ローカルLLM接続テスト ===")
    
    # gemma2:2bモデルでテスト（日本語により適している）
    client = LocalLLMClient(model="gemma2:2b")
    
    if not client.ollama_available:
        print("❌ Ollamaが利用できません")
        return False
    
    print("✅ Ollama接続成功")
    
    # 簡単なテスト
    test_text = "物理学の基本概念について"
    print(f"\nテスト入力: {test_text}")
    print("処理中...")
    
    result = client.summarize_text(test_text)
    print(f"\n結果:\n{result}")
    
    return True

if __name__ == "__main__":
    test_local_llm()
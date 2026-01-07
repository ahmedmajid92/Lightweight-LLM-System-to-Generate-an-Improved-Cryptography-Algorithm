# src/chat_orchestrator.py
from typing import List, Dict, Any
# Reuse your existing RAG chat without touching it
from .enhanced_rag_chat import EnhancedRAGChat  # assuming this is your class

class ChatOrchestrator:
    def __init__(self):
        self.rag = EnhancedRAGChat()

    def chat(self, history: List[List[str]], user_msg: str) -> (List[List[str]], str):
        """
        Gradio-style: history is [[user, assistant], ...]
        """
        result = self.rag.chat(user_msg)  # returns a dict with 'response' key
        reply = result['response']
        history = history or []
        history.append([user_msg, reply])
        return history, reply

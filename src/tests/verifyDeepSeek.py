"""
Test script for the new DeepSeek Coder integration.
"""

from enhanced_rag_chat import EnhancedRAGChat

def test_model():
    """Test the DeepSeek Coder model integration."""
    
    print("🧪 Testing DeepSeek Coder Integration")
    print("=" * 40)
    
    try:
        # Initialize the RAG system
        print("Initializing RAG system...")
        rag = EnhancedRAGChat()
        
        # Test queries
        test_queries = [
            "Show me AES encryption implementation",
            "Recommend a cipher for high security applications",
            "How does DES key schedule work?",
            "Give me Blowfish components only"
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 30)
            
            result = rag.chat(query)
            
            print(f"🎯 Intent: {result['intent']}")
            print(f"🔧 Method: {result['method']}")
            print(f"📊 Chunks: {result['retrieved_chunks']}")
            
            # Show brief response
            response = result['response']
            if len(response) > 200:
                response = response[:200] + "..."
            print(f"🤖 Response: {response}")
            
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("💡 Make sure you have sufficient GPU memory for the 7B model")

if __name__ == "__main__":
    test_model()
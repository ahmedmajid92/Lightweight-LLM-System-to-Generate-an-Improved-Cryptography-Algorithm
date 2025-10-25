"""
Utility script to update embeddings with a new model.
This will regenerate all embeddings using CodeBERTa.
"""

import os
from generate_embeddings import main as generate_main

def update_embeddings():
    """Update embeddings to use CodeBERTa model."""
    
    print("🔄 Updating embeddings to use CodeBERTa-small-v1...")
    print("This will regenerate all embeddings and may take a few minutes.\n")
    
    # Check if old embeddings exist
    old_file = 'data/algorithm_embeddings.pkl'
    if os.path.exists(old_file):
        backup_file = 'data/algorithm_embeddings_backup.pkl'
        print(f"📁 Backing up existing embeddings to {backup_file}")
        os.rename(old_file, backup_file)
    
    # Generate new embeddings
    print("🚀 Generating new embeddings with CodeBERTa...")
    generate_main()
    
    print("\n✅ Embeddings updated successfully!")
    print("🔍 CodeBERTa should provide better understanding of code-related queries.")
    
    # Test the new embeddings
    print("\n🧪 Testing new embeddings...")
    from enhanced_rag_chat import EnhancedRAGChat
    
    try:
        rag = EnhancedRAGChat()
        
        test_queries = [
            "Show me AES mix columns implementation",
            "How does DES key schedule work", 
            "Give me Blowfish F function code"
        ]
        
        for query in test_queries:
            result = rag.chat(query)
            print(f"\n📝 Query: {query}")
            print(f"🎯 Intent: {result['intent']}")
            print(f"📊 Retrieved chunks: {result['retrieved_chunks']}")
            if result['chunk_details']:
                best_match = result['chunk_details'][0]
                print(f"🏆 Best match: {best_match['id']} (similarity: {best_match['similarity']:.3f})")
        
        print("\n🎉 New embedding system is working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing new embeddings: {e}")
        print("You may need to install missing dependencies or check your model paths.")

if __name__ == "__main__":
    update_embeddings()
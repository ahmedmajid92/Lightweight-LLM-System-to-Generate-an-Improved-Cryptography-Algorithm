import gradio as gr
import json
from enhanced_rag_chat import EnhancedRAGChat
import threading
import time

class WebRAGChat:
    def __init__(self):
        self.rag = None
        self.initialization_status = "Not started"
        self.error_message = ""
        
    def initialize_rag(self):
        """Initialize the RAG system in a separate thread to avoid blocking the UI."""
        try:
            self.initialization_status = "Loading embedding database..."
            yield self.initialization_status
            
            self.rag = EnhancedRAGChat()
            self.initialization_status = "✅ RAG system ready!"
            yield self.initialization_status
            
        except Exception as e:
            self.error_message = f"❌ Error initializing RAG system: {str(e)}"
            self.initialization_status = self.error_message
            yield self.initialization_status

    def chat_interface(self, message, history):
        """Handle chat messages and return responses."""
        if not self.rag:
            # Return proper message format
            return history + [{"role": "assistant", "content": "Please wait for the system to initialize first."}]
        
        if not message.strip():
            return history
        
        try:
            # Process the query
            result = self.rag.chat(message)
            
            # Format the response
            if result['intent'] == 'code':
                response = f"""**🧠 Intent:** {result['intent']}  
**🔧 Method:** {result['method']}  
**📊 Retrieved:** {result['retrieved_chunks']} relevant chunks

**💻 Code Response:**
```python
{result['response']}
```

**📚 Sources:**"""
                
                # Add sources
                if result['chunk_details']:
                    for chunk in result['chunk_details'][:3]:
                        response += f"\n- {chunk['id']} ({chunk['type']}, similarity: {chunk['similarity']:.3f})"
            
            else:
                response = f"""**🧠 Intent:** {result['intent']}  
**🔧 Method:** {result['method']}  
**📊 Retrieved:** {result['retrieved_chunks']} relevant chunks

**🤖 Response:**
{result['response']}

**📚 Sources:**"""
                
                # Add sources
                if result['chunk_details']:
                    for chunk in result['chunk_details'][:3]:
                        response += f"\n- {chunk['id']} ({chunk['type']}, similarity: {chunk['similarity']:.3f})"
            
            # Add to history using new message format
            new_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
            return new_history
            
        except Exception as e:
            error_response = f"❌ Error processing query: {str(e)}"
            new_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": error_response}
            ]
            return new_history

def create_interface():
    """Create and configure the Gradio interface."""
    
    # Initialize the web chat system
    web_chat = WebRAGChat()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .chatbot {
        height: 600px !important;
    }
    .chat-message {
        padding: 10px !important;
    }
    """
    
    with gr.Blocks(css=css, title="🔐 Cryptography RAG Chat") as app:
        gr.Markdown("""
        # 🔐 Enhanced Cryptography RAG Chat
        
        Ask me about cryptographic algorithms, implementations, or get recommendations!
        
        **Examples:**
        - "Show me AES mix columns implementation"
        - "Give me full Blowfish implementation" 
        - "Recommend a cipher for high security"
        - "Components only for DES"
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # System status panel
                gr.Markdown("### 🚀 System Status")
                status_display = gr.Textbox(
                    label="Initialization Status",
                    value="Click 'Initialize System' to start",
                    interactive=False,
                    lines=3
                )
                
                init_btn = gr.Button("🔄 Initialize System", variant="primary")
                
                # Quick action buttons
                gr.Markdown("### 🎯 Quick Actions")
                
                with gr.Row():
                    aes_btn = gr.Button("AES Mix Columns", size="sm")
                    des_btn = gr.Button("DES Components", size="sm")
                
                with gr.Row():
                    blowfish_btn = gr.Button("Full Blowfish", size="sm")
                    recommend_btn = gr.Button("Recommend Cipher", size="sm")
                
                # Algorithm info panel
                gr.Markdown("### 📚 Available Algorithms")
                gr.Markdown("""
                **Block Ciphers:**
                - AES (Advanced Encryption Standard)
                - DES (Data Encryption Standard)  
                - 3DES (Triple DES)
                - Blowfish
                - Twofish
                - IDEA
                - RC5
                - Camellia
                - CAST-128
                - Serpent
                - MARS
                - RC6
                """)
            
            with gr.Column(scale=2):
                # Main chat interface - SET type='messages' explicitly
                chatbot = gr.Chatbot(
                    label="💬 Chat with Crypto Expert",
                    height=600,
                    show_label=True,
                    container=True,
                    bubble_full_width=False,
                    type="messages"  # ADD THIS LINE
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about cryptographic algorithms, implementations, or recommendations...",
                        label="Your Message",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send 📤", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    retry_btn = gr.Button("🔄 Retry Last", variant="secondary")
        
        # Event handlers
        def init_system():
            """Initialize the RAG system with progress updates."""
            for status in web_chat.initialize_rag():
                yield status
        
        def send_message(message, history):
            """Send message and get response."""
            return web_chat.chat_interface(message, history), ""
        
        def clear_chat():
            """Clear the chat history."""
            return []
        
        def quick_action(query, history):
            """Handle quick action buttons."""
            return web_chat.chat_interface(query, history)
        
        # Wire up the events
        init_btn.click(
            fn=init_system,
            outputs=[status_display]
        )
        
        send_btn.click(
            fn=send_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            fn=send_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot]
        )
        
        # Quick action buttons
        aes_btn.click(
            fn=lambda history: quick_action("Show me AES mix columns implementation", history),
            inputs=[chatbot],
            outputs=[chatbot]
        )
        
        des_btn.click(
            fn=lambda history: quick_action("Components only for DES", history),
            inputs=[chatbot],
            outputs=[chatbot]
        )
        
        blowfish_btn.click(
            fn=lambda history: quick_action("Give me full Blowfish implementation", history),
            inputs=[chatbot],
            outputs=[chatbot]
        )
        
        recommend_btn.click(
            fn=lambda history: quick_action("Recommend a cipher for high security", history),
            inputs=[chatbot],
            outputs=[chatbot]
        )
        
        # Add footer
        gr.Markdown("""
        ---
        **💡 Tips:**
        - Use specific component names like "mix columns", "key expansion", "S-box"
        - Ask for "full implementation" to get complete algorithm code
        - Request "components only" to see just the helper functions
        - Ask for recommendations based on your security requirements
        """)
    
    return app

def main():
    """Launch the web interface."""
    print("🚀 Starting Cryptography RAG Chat Web Interface...")
    
    # Create and launch the interface
    app = create_interface()
    
    app.launch(
        share=False,              # Set to True to create a public link
        debug=False,              # Set to True for development
        show_error=True,          # Show errors in the interface
        inbrowser=True            # Automatically open in browser
    )

if __name__ == "__main__":
    main()
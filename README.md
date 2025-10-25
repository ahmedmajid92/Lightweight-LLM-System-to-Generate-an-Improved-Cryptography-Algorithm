# 🔐 Cryptography RAG Chat

A Retrieval-Augmented Generation (RAG) system for cryptographic algorithm assistance, powered by DeepSeek Coder 7B and featuring an intuitive web interface.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)

## 🎯 Overview

This intelligent chat system provides instant access to cryptographic algorithm implementations, explanations, and recommendations. Built with a RAG architecture, it combines semantic search with LLM-powered generation to deliver accurate, code-level responses without hallucination.

### Key Features

- 🧠 **Smart Intent Detection** - Automatically understands if you want code, recommendations, or explanations
- 💻 **Direct Code Retrieval** - Returns actual implementation code from source files (zero hallucination)
- 🤖 **LLM-Powered Q&A** - Uses DeepSeek Coder 7B for intelligent responses to complex queries
- 🎨 **Beautiful Web Interface** - Built with Gradio for seamless interaction
- 📚 **12 Algorithms Supported** - AES, DES, 3DES, Blowfish, Twofish, Serpent, Camellia, CAST-128, IDEA, RC5, RC6, SEED
- ⚡ **Fast Semantic Search** - CodeBERTa embeddings for relevant chunk retrieval

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Interface (Gradio)                    │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced RAG Chat System                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ Intent Classifier│  │ Semantic Search  │  │ Code Retrieval│ │
│  │  • code          │  │  • CodeBERTa     │  │ • Components  │ │
│  │  • recommendation│──▶│  • FAISS index   │──▶│ • Algorithms  │ │
│  │  • general       │  │  • Top-K chunks  │  │ • Direct load │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
│                                  │                               │
│                                  ▼                               │
│                    ┌─────────────────────────┐                  │
│                    │   LLM Generation        │                  │
│                    │   DeepSeek Coder 7B     │                  │
│                    │   (4-bit quantized)     │                  │
│                    └─────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU with 6GB+ VRAM (recommended)
  - CPU mode supported but slower
- **Disk Space**: ~10GB for models
- **RAM**: 8GB minimum, 16GB recommended

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd My-Project
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python src/launch_web.py
```

The web interface will automatically open at `http://localhost:7860`

On first run, the system will download the DeepSeek Coder 7B model (~7GB). This may take several minutes depending on your internet connection.

## 💡 Usage Examples

### Get Component Code
```
Show me AES mix columns implementation
```
Returns the actual `mix_columns()` function from `Components.py`

### Get Full Algorithm Implementation
```
Give me full Blowfish implementation
```
Returns complete encryption/decryption functions with all components

### Algorithm Recommendations
```
Recommend a cipher for high security applications
```
LLM analyzes your requirements and suggests the best algorithm

### Components Only
```
Components only for DES
```
Returns just the helper functions (key schedule, F-function, permutations)

### Ask Questions
```
What is the difference between Feistel and SPN structure?
```
LLM provides detailed explanations using retrieved context

## 📁 Project Structure

```
My-Project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/
│   ├── launch_web.py                  # Entry point
│   ├── web_ui.py                      # Gradio web interface
│   ├── enhanced_rag_chat.py           # RAG system core
│   ├── Components.py                  # Cryptographic component functions
│   ├── AlgorithmsBlock.py             # Main encrypt/decrypt implementations
│   ├── tests/                         # Test files
│   │   ├── test_crypto.py             # Algorithm tests
│   │   └── ...
│   └── tools/                         # Build and setup utilities
│       ├── build_embeddings.py        # Generate embeddings
│       ├── generate_embeddings.py     # Alternative embedding generator
│       └── ...
├── data/
│   ├── algorithms.json                # Algorithm metadata
│   ├── algorithm_embeddings.pkl       # Precomputed embeddings
│   └── algorithm_implementations.json # Implementation mappings
└── models/                            # Downloaded models (auto-created)
```

## 🔧 Technical Details

### RAG Pipeline

1. **Query Analysis**: Classify user intent (code/recommendation/general)
2. **Semantic Search**: Find relevant chunks using CodeBERTa embeddings
3. **Smart Routing**:
   - **Code queries** → Direct file retrieval from source
   - **Recommendations** → LLM analysis with retrieved context
   - **General queries** → LLM generation with retrieved context

### Models

| Component | Model | Purpose |
|-----------|-------|---------|
| Embeddings | `huggingface/CodeBERTa-small-v1` | Semantic search over code/docs |
| LLM | `deepseek-ai/deepseek-coder-7b-instruct-v1.5` | Q&A and recommendations |
| Quantization | 4-bit NF4 | Memory efficiency (~6GB VRAM) |

### Supported Algorithms

| Algorithm | Year | Structure | Block Size | Security | Speed | Applications |
|-----------|------|-----------|------------|----------|-------|--------------|
| **AES** | 2001 | SPN | 128-bit | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | TLS, IPsec, Disk encryption |
| **DES** | 1977 | Feistel | 64-bit | ⭐ | ⭐ | Legacy systems |
| **3DES** | 1998 | Feistel | 64-bit | ⭐⭐ | ⭐⭐ | Banking, Smart cards |
| **Blowfish** | 1993 | Feistel | 64-bit | ⭐⭐⭐ | ⭐⭐⭐⭐ | OpenVPN, bcrypt |
| **Twofish** | 1998 | Feistel | 128-bit | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | TrueCrypt, Disk encryption |
| **Serpent** | 1998 | SPN | 128-bit | ⭐⭐⭐⭐⭐ | ⭐⭐ | High-security applications |
| **Camellia** | 2000 | Feistel | 128-bit | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | TLS, Japanese standards |
| **CAST-128** | 1996 | Feistel | 64-bit | ⭐⭐⭐ | ⭐⭐⭐ | PGP, IPsec |
| **IDEA** | 1991 | Lai-Massey | 64-bit | ⭐⭐⭐⭐ | ⭐⭐⭐ | PGP (historical) |
| **RC5** | 1994 | Feistel | Variable | ⭐⭐⭐ | ⭐⭐⭐ | Research, Custom protocols |
| **RC6** | 1998 | Feistel | 128-bit | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | AES candidate |
| **SEED** | 1998 | Feistel | 128-bit | ⭐⭐⭐⭐ | ⭐⭐⭐ | Korean standards |

## 🎮 Web Interface Features

- **Quick Action Buttons**: One-click queries for common requests
- **Real-time Status**: Live system initialization progress
- **Syntax Highlighting**: Beautiful code rendering in responses
- **Source Attribution**: Shows which chunks were used for each response
- **Chat History**: Maintains conversation context
- **Responsive Design**: Works on desktop and mobile

## 🔬 Development

### Running Tests

```bash
# Test algorithm implementations
python src/tests/test_crypto.py

# Test RAG system
python src/verifyDeepSeek.py
```

### Building Embeddings

If you modify the algorithm implementations or add new algorithms:

```bash
python src/tools/build_embeddings.py
```

This will regenerate the embedding database.

## ⚡ Performance

| Operation | Time (GPU) | Time (CPU) |
|-----------|------------|------------|
| System initialization | ~30s | ~60s |
| Embedding lookup | 50ms | 100ms |
| Code retrieval | 100ms | 100ms |
| LLM generation | 2-5s | 20-60s |

**First query is slower** due to model loading. Subsequent queries are much faster.

## 🐛 Troubleshooting

### Out of Memory Error
```python
# The model uses 4-bit quantization to minimize memory
# Requires: ~6GB GPU VRAM or ~12GB system RAM
# Solution: Close other applications or use a machine with more RAM
```

### Model Download Issues
```bash
# Models are cached in ~/.cache/huggingface/
# To clear cache and re-download:
rm -rf ~/.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5
```

### Slow Generation
- **Use GPU**: Ensure CUDA is properly installed
- **Check device**: The system should show `cuda:0` during initialization
- **First query slow**: Model loading takes time, subsequent queries are faster

### Import Errors
```bash
# Ensure you're in the virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Add more algorithms** (ARIA, SM4, etc.)
- **Improve component extraction** (better parsing of function dependencies)
- **Performance optimizations** (caching, batching)
- **UI enhancements** (dark mode, export functionality)
- **Additional features** (algorithm comparison, security analysis)

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `python src/tests/test_crypto.py`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📝 License

This project is for educational purposes. Individual algorithm implementations follow their respective specifications and standards.

## 🙏 Acknowledgments

- **DeepSeek AI** - For the excellent DeepSeek Coder model
- **Hugging Face** - For model hosting and transformers library
- **Gradio** - For the beautiful web interface framework
- **Cryptographic Community** - For algorithm specifications and research

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**⚠️ Important Note**: This is an educational project. For production cryptographic systems, use well-tested libraries like `cryptography`, `PyCrypto`, or `OpenSSL`. Never implement your own cryptography for real-world security applications.

## 🌟 Star History

If you find this project helpful, please consider giving it a star ⭐

---

Made with ❤️ for the cryptography community


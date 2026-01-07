# 🔐 Lightweight LLM System for Cryptography Algorithm Generation

A comprehensive dual-interface system combining **AI-powered cryptography assistance** with **dynamic algorithm composition**. Built with DeepSeek Coder 7B for intelligent Q&A and featuring an innovative component-based algorithm builder.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

## 🎯 Overview

This system provides two powerful interfaces for working with cryptographic algorithms:

### 1️⃣ **RAG Chat Interface** - Intelligent Cryptography Assistant

A Retrieval-Augmented Generation (RAG) system that provides instant access to cryptographic algorithm implementations, explanations, and recommendations. Built with a RAG architecture combining semantic search with LLM-powered generation for accurate, code-level responses without hallucination.

### 2️⃣ **Component Composer Interface** - Dynamic Algorithm Builder

An innovative system for creating custom cryptographic algorithms by mixing and matching components from different ciphers. Select your base algorithm, choose components from any cipher, and generate working implementations with automated validation.

---

## ✨ Key Features

### RAG Chat Interface

- 🧠 **Smart Intent Detection** - Automatically understands if you want code, recommendations, or explanations
- 💻 **Direct Code Retrieval** - Returns actual implementation code from source files (zero hallucination)
- 🤖 **LLM-Powered Q&A** - Uses DeepSeek Coder 7B for intelligent responses to complex queries
- 🎨 **Beautiful Web Interface** - Built with Gradio for seamless interaction
- 📚 **12 Algorithms Supported** - AES, DES, 3DES, Blowfish, Twofish, Serpent, Camellia, CAST-128, IDEA, RC5, RC6, SEED
- ⚡ **Fast Semantic Search** - CodeBERTa embeddings for relevant chunk retrieval

### Component Composer Interface

- 🔧 **Mix & Match Components** - Combine encryption stages from different algorithms (e.g., AES SubBytes with Serpent MixColumns)
- 🏗️ **Blueprint-Based Generation** - Uses algorithm blueprints (SPN, Feistel) to ensure structural correctness
- ✅ **Automated Validation** - Comprehensive testing including round-trip encryption/decryption
- 📊 **Component Browsing** - Explore all available components with their signatures and documentation
- 💾 **Export Generated Algorithms** - Save your custom algorithms as Python modules
- 🔍 **Smart Component Filtering** - Filter components by algorithm, stage, or search terms
- 🎯 **Composition Chat** - Get AI assistance while building your algorithms

---

## 🏗️ Architecture

### RAG Chat System

```
┌─────────────────────────────────────────────────────────────────┐
│                    Web Interface (Gradio)                        │
│                     http://localhost:7860                        │
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
│                                 │                               │
│                                 ▼                               │
│                    ┌─────────────────────────┐                  │
│                    │   LLM Generation        │                  │
│                    │   DeepSeek Coder 7B     │                  │
│                    │   (4-bit quantized)     │                  │
│                    └─────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Composer System

```
┌─────────────────────────────────────────────────────────────────┐
│              Component Composer Web Interface                    │
│                   http://localhost:7861                          │
└─────────────────────────────────────────────────────────────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 ▼                ▼                ▼
        ┌────────────────┐ ┌────────────┐ ┌──────────────┐
        │ Component      │ │ Algorithm  │ │ Composer     │
        │ Registry       │ │ Blueprints │ │ Engine       │
        │ • Scan         │ │ • SPN      │ │ • Code Gen   │
        │ • Classify     │ │ • Feistel  │ │ • Assembly   │
        │ • Filter       │ │ • Stages   │ │ • Validation │
        └────────────────┘ └────────────┘ └──────────────┘
                 │                │                │
                 └────────────────┼────────────────┘
                                  ▼
                       ┌──────────────────────┐
                       │  Generated Module    │
                       │  • encrypt_block()   │
                       │  • decrypt_block()   │
                       │  • Auto-validated    │
                       └──────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU with 6GB+ VRAM (recommended for RAG Chat)
  - CPU mode supported but slower
- **Disk Space**: ~10GB for models (RAG Chat only)
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

### Running the Applications

#### Option 1: RAG Chat Interface (Port 7860)

```bash
python src/launch_web.py
```

The RAG Chat interface will open at `http://localhost:7860`

On first run, the system will download the DeepSeek Coder 7B model (~7GB). This may take several minutes.

#### Option 2: Component Composer Interface (Port 7861)

```bash
python src/web_ui_v2.py
```

The Component Composer interface will open at `http://localhost:7861`

No model download required - works immediately!

---

## 💡 Usage Examples

### RAG Chat Interface

#### Get Component Code

```
Show me AES mix columns implementation
```

Returns the actual `mix_columns()` function from `Components.py`

#### Get Full Algorithm Implementation

```
Give me full Blowfish implementation
```

Returns complete encryption/decryption functions with all components

#### Algorithm Recommendations

```
Recommend a cipher for high security applications
```

LLM analyzes your requirements and suggests the best algorithm

#### Components Only

```
Components only for DES
```

Returns just the helper functions (key schedule, F-function, permutations)

#### Ask Questions

```
What is the difference between Feistel and SPN structure?
```

LLM provides detailed explanations using retrieved context

### Component Composer Interface

#### Workflow

1. **Browse Components** - Explore all cryptographic components from `Components.py` with filtering and search
2. **Select Components** - Check components you want to use in your custom algorithm
3. **Choose Base Algorithm** - Select the structural blueprint (AES, Serpent, DES, etc.)
4. **Map Components to Stages** - Assign selected components to algorithm stages (SubBytes, MixColumns, etc.)
5. **Generate Algorithm** - Click "Generate Composed Algorithm" to create your custom cipher
6. **Validate** - Automatically test encryption/decryption round-trip with random data
7. **Export** - Save the generated module to `data/generated_algorithms/`

#### Example: Custom SPN Cipher

- Base: AES (SPN structure, 10 rounds)
- SubBytes: Use Serpent's S-box layer
- ShiftRows: Keep AES shift rows
- MixColumns: Use Twofish's mixing function
- KeySchedule: Use AES key expansion

The system generates a working algorithm with proper inverse operations for decryption!

---

## 📁 Project Structure

```
My-Project/
├── README.md                          # This comprehensive guide
├── requirements.txt                   # Python dependencies
│
├── src/                               # Source code
│   ├── launch_web.py                  # RAG Chat entry point (Port 7860)
│   ├── web_ui_v2.py                   # Component Composer entry point (Port 7861)
│   │
│   ├── # RAG Chat System
│   ├── web_ui.py                      # RAG Chat Gradio interface
│   ├── enhanced_rag_chat.py           # RAG system core (576 lines)
│   ├── chat_orchestrator.py           # Chat session management
│   │
│   ├── # Component Composer System
│   ├── web_ui_components.py           # Composer Gradio interface (441 lines)
│   ├── composer.py                    # Algorithm composition engine (238 lines)
│   ├── algorithm_blueprints.py        # Blueprint definitions (SPN, Feistel)
│   ├── component_registry.py          # Component scanning & classification (158 lines)
│   ├── validator.py                   # Generated algorithm validator (182 lines)
│   ├── schemas.py                     # Data schemas for composition
│   │
│   ├── # Cryptographic Implementations
│   ├── Components.py                  # 91 cryptographic functions (1303 lines)
│   ├── AlgorithmsBlock.py             # 12 algorithm implementations (485 lines)
│   │
│   ├── tests/                         # Test suite
│   │   ├── test_crypto.py             # Algorithm correctness tests
│   │   ├── test_composer_validator.py # Composition system tests
│   │   ├── test_rc5_rc6_components.py # RC5/RC6 specific tests
│   │   └── verifyDeepSeek.py          # RAG system verification
│   │
│   └── tools/                         # Build utilities
│       ├── build_embeddings.py        # Generate semantic embeddings
│       ├── generate_embeddings.py     # Advanced embedding generator (12KB)
│       ├── build_finetune_dataset.py  # Create training datasets
│       └── update_embeddings.py       # Incremental embedding updates
│
├── data/                              # Data files
│   ├── algorithms.json                # Algorithm metadata (242 lines)
│   ├── algorithm_implementations.json # Implementation mappings (210 lines)
│   ├── algorithm_embeddings.pkl       # Precomputed embeddings (208KB)
│   ├── algos.index                    # FAISS index for semantic search
│   ├── block_cipher_dataset.jsonl     # Training dataset (3.2MB)
│   └── generated_algorithms/          # Your composed algorithms
│
└── models/                            # Downloaded models (auto-created)
    └── .gitkeep                       # Models cached here (~7GB)
```

---

## 🔧 Technical Details

### RAG Chat Pipeline

1. **Query Analysis**: Classify user intent using keyword matching (code/recommendation/general)
2. **Semantic Search**: Find relevant code chunks using CodeBERTa embeddings and FAISS
3. **Smart Routing**:
   - **Code queries** → Direct file retrieval from `Components.py` or `AlgorithmsBlock.py`
   - **Recommendations** → LLM analysis with retrieved context
   - **General queries** → LLM generation with retrieved context
4. **Response Generation**: DeepSeek Coder 7B generates human-readable responses with code snippets

### Component Composer Pipeline

1. **Component Scanning**: Introspect `Components.py` using Python's `inspect` module
2. **Classification**: Heuristic-based algorithm and stage detection (prefix matching, docstrings, known function mapping)
3. **Blueprint Selection**: Load structural template (SPN/Feistel) with stage definitions
4. **Code Generation**: Generate Python module with:
   - `encrypt_block(plaintext: bytes, key: bytes) -> bytes`
   - `decrypt_block(ciphertext: bytes, key: bytes) -> bytes`
   - Proper round loops, inverse component selection
5. **Validation**: Round-trip testing with random plaintexts matching declared block/key sizes
6. **Export**: Save to `data/generated_algorithms/{algorithm_name}.py`

### Models & Technologies

| Component         | Technology                                    | Purpose                 | Size      |
| ----------------- | --------------------------------------------- | ----------------------- | --------- |
| **Embeddings**    | `huggingface/CodeBERTa-small-v1`              | Semantic code search    | ~100MB    |
| **LLM**           | `deepseek-ai/deepseek-coder-7b-instruct-v1.5` | Q&A and recommendations | ~7GB      |
| **Quantization**  | 4-bit NF4 (bitsandbytes)                      | Memory efficiency       | ~6GB VRAM |
| **Search Index**  | FAISS                                         | Fast similarity search  | 18KB      |
| **Web UI**        | Gradio 4.0+                                   | Interactive interfaces  | -         |
| **Code Analysis** | Python `inspect`                              | Component introspection | -         |

### Supported Algorithms

#### Encryption Algorithms (12 Total)

| Algorithm    | Year | Structure    | Block Size | Key Size    | Rounds   | Security   | Speed      | Applications                |
| ------------ | ---- | ------------ | ---------- | ----------- | -------- | ---------- | ---------- | --------------------------- |
| **AES**      | 2001 | SPN          | 128-bit    | 128/192/256 | 10/12/14 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | TLS, IPsec, Disk encryption |
| **DES**      | 1977 | Feistel      | 64-bit     | 56          | 16       | ⭐         | ⭐         | Legacy systems              |
| **3DES**     | 1998 | Feistel      | 64-bit     | 168         | 48       | ⭐⭐       | ⭐⭐       | Banking, Smart cards        |
| **Blowfish** | 1993 | Feistel      | 64-bit     | 32-448      | 16       | ⭐⭐⭐     | ⭐⭐⭐⭐   | OpenVPN, bcrypt             |
| **Twofish**  | 1998 | Feistel-like | 128-bit    | 128/192/256 | 16       | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | TrueCrypt, Disk encryption  |
| **Serpent**  | 1998 | SPN          | 128-bit    | 128/192/256 | 32       | ⭐⭐⭐⭐⭐ | ⭐⭐       | High-security applications  |
| **Camellia** | 2000 | Feistel-like | 128-bit    | 128/192/256 | 18/24    | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | TLS, Japanese standards     |
| **CAST-128** | 1996 | Feistel      | 64-bit     | 40-128      | 12/16    | ⭐⭐⭐     | ⭐⭐⭐     | PGP, IPsec                  |
| **IDEA**     | 1991 | Lai-Massey   | 64-bit     | 128         | 8.5      | ⭐⭐⭐⭐   | ⭐⭐⭐     | PGP (historical)            |
| **RC5**      | 1994 | Feistel-like | 32/64/128  | up to 2048  | Variable | ⭐⭐⭐     | ⭐⭐⭐     | Research, Custom protocols  |
| **RC6**      | 1998 | Feistel-like | 128-bit    | 128/192/256 | 20       | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | AES candidate               |
| **SEED**     | 1998 | Feistel      | 128-bit    | 128         | 16       | ⭐⭐⭐⭐   | ⭐⭐⭐     | Korean standards            |

#### Component Categories (91 Functions)

| Category                       | Count | Examples                                                        |
| ------------------------------ | ----- | --------------------------------------------------------------- |
| **Utility Functions**          | 15    | `bytes_to_int`, `rotate_left`, `xor_bytes`, `apply_permutation` |
| **SPN Components**             | 12    | `apply_sbox_layer`, `spn_encrypt`, AES operations               |
| **Feistel Components**         | 8     | `feistel_round`, `feistel_cipher`, F-functions                  |
| **AES Functions**              | 10    | `sub_bytes`, `shift_rows`, `mix_columns`, `aes_key_expansion`   |
| **DES Functions**              | 8     | `des_f`, `des_key_schedule`, permutations                       |
| **Blowfish Functions**         | 6     | `blowfish_f`, `blowfish_key_schedule`                           |
| **Twofish Functions**          | 8     | `twofish_f`, `twofish_key_schedule`, `twofish_h`                |
| **Serpent Functions**          | 4     | `serpent_key_schedule`, S-boxes                                 |
| **Other Algorithm Components** | 30    | Camellia, CAST-128, IDEA, RC5, RC6, SEED functions              |

---

## 🎮 Web Interface Features

### RAG Chat Interface (Port 7860)

- ✨ **Quick Action Buttons** - One-click common queries
- 📊 **Real-time Status** - Live initialization progress
- 🎨 **Syntax Highlighting** - Beautiful code rendering
- 📚 **Source Attribution** - Shows retrieved chunks
- 💬 **Chat History** - Maintains conversation context
- 📱 **Responsive Design** - Desktop and mobile support

### Component Composer Interface (Port 7861)

- 🔍 **Component Browser** - Searchable component table with 91 functions
- 🏷️ **Smart Filtering** - Filter by algorithm, stage, or keyword
- ✅ **Checkbox Selection** - Multi-select components with count display
- 📋 **Stage Mapping** - Dropdown assignment of components to stages
- 🎯 **Blueprint Selection** - Choose from 12 base algorithms
- ✅ **Live Validation** - Real-time feedback on composition issues
- 💬 **Composition Chat** - AI assistant for guidance
- 💾 **Export & Download** - Save generated algorithms

---

## 🔬 Development

### Running Tests

```bash
# Test all 12 algorithm implementations
python src/tests/test_crypto.py

# Test RAG system functionality
python src/tests/verifyDeepSeek.py

# Test composer and validator
python src/tests/test_composer_validator.py

# Test RC5/RC6 specific components
python src/tests/test_rc5_rc6_components.py
```

### Building Embeddings

If you modify algorithm implementations or add new algorithms:

```bash
# Quick rebuild (FAISS index only)
python src/tools/build_embeddings.py

# Full rebuild (comprehensive embeddings)
python src/tools/generate_embeddings.py

# Incremental update
python src/tools/update_embeddings.py
```

### Adding New Components

1. **Add function to `Components.py`**:

```python
def my_new_component(state: int, key: int) -> int:
    """
    Algorithm: MyAlgorithm
    Stage: transformation

    Custom transformation logic.
    """
    return state ^ key  # Example
```

2. **Update `algorithm_blueprints.py`** if creating new structure

3. **Rebuild embeddings**:

```bash
python src/tools/build_embeddings.py
```

4. **Test the component**:

```python
from src.component_registry import scan_components
comps = scan_components()
# Verify your component is discovered
```

### Creating Custom Algorithms

See `src/tests/test_composer_validator.py` for programmatic examples:

```python
from src.composer import compose
from src.algorithm_blueprints import get_blueprint
from src.schemas import CompositionRequest

blueprint = get_blueprint("AES")
request = CompositionRequest(
    base_algorithm="AES",
    selections={
        "sub_bytes": "serpent_sbox_0",
        "shift_rows": "shift_rows",
        "mix_columns": "twofish_mds_multiply",
        "key_schedule": "aes_key_expansion"
    },
    output_name="MyCustomCipher"
)

result = compose(blueprint, request)
print(result.module_code)  # View generated code
```

---

## ⚡ Performance

### RAG Chat System

| Operation             | Time (GPU) | Time (CPU) |
| --------------------- | ---------- | ---------- |
| System initialization | ~30s       | ~60s       |
| Embedding lookup      | 50ms       | 100ms      |
| Code retrieval        | 100ms      | 100ms      |
| LLM generation        | 2-5s       | 20-60s     |

**Note**: First query is slower due to model loading (~10-15s). Subsequent queries are much faster.

### Component Composer System

| Operation                    | Time   |
| ---------------------------- | ------ |
| Component scanning           | ~500ms |
| Code generation              | ~200ms |
| Validation (round-trip test) | ~100ms |
| Total composition time       | ~1s    |

**Note**: No model loading required - instant startup!

---

## 🐛 Troubleshooting

### RAG Chat Issues

#### Out of Memory Error

```python
# The model uses 4-bit quantization to minimize memory
# Requires: ~6GB GPU VRAM or ~12GB system RAM
# Solutions:
# 1. Close other applications
# 2. Use a machine with more RAM
# 3. Run on CPU (slower but works with less RAM)
```

#### Model Download Issues

```bash
# Models are cached in ~/.cache/huggingface/
# To clear cache and re-download:
# Linux/Mac:
rm -rf ~/.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5

# Windows:
rmdir /S %USERPROFILE%\.cache\huggingface\hub\models--deepseek-ai--deepseek-coder-7b-instruct-v1.5
```

#### Slow Generation

- **Use GPU**: Ensure CUDA is properly installed (`torch.cuda.is_available()` should return `True`)
- **Check device**: System should show `Using device: cuda:0` during initialization
- **First query slow**: Model loading takes ~10-15s, subsequent queries are 2-5s

### Component Composer Issues

#### Component Not Found

```python
# Rebuild component registry
from src.component_registry import scan_components
comps = scan_components()
print(list(comps.keys()))  # View all discovered algorithms
```

#### Validation Failed

- **Check block/key sizes**: Ensure they match component expectations
- **Review stage mappings**: Some components are incompatible with certain structures
- **Check error messages**: Validation provides detailed structured feedback

#### Generated Code Errors

- **Import errors**: Ensure all referenced components exist in `Components.py`
- **Signature mismatches**: Check component signatures match expected inputs/outputs
- **Round trip failure**: Components may not have proper inverse operations

### General Issues

#### Import Errors

```bash
# Ensure you're in the virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Verify installation
python -c "import torch; import gradio; import transformers; print('OK')"
```

#### Port Already in Use

```bash
# If port 7860 or 7861 is already in use:
# Option 1: Kill the process
# Windows: netstat -ano | findstr :7860
# Linux: lsof -i :7860

# Option 2: Change the port in the code
# Edit web_ui.py or web_ui_v2.py and modify server_port parameter
```

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

### Algorithm Implementation

- Add more ciphers (ARIA, SM4, ChaCha20, etc.)
- Implement additional modes (CBC, CTR, GCM)
- Add stream ciphers support

### Component Composer

- Support for hybrid structures (SPN + Feistel)
- Performance benchmarking of composed algorithms
- Visual composition graph/flowchart
- Export to C/C++ code

### RAG Chat

- Improve intent classification with ML
- Add algorithm comparison features
- Security analysis recommendations
- Code vulnerability detection

### UI/UX

- Dark mode support
- Export conversations as PDF
- Live code editor with syntax highlighting
- Mobile app version

### Performance

- Caching for frequently requested code
- Batched embedding generation
- Quantization optimization
- Distributed computation for validation

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes** with clear commit messages
4. **Run tests**:
   ```bash
   python src/tests/test_crypto.py
   python src/tests/test_composer_validator.py
   ```
5. **Update documentation** if needed
6. **Commit your changes**:
   ```bash
   git commit -m 'Add amazing feature: description'
   ```
7. **Push to your fork**:
   ```bash
   git push origin feature/amazing-feature
   ```
8. **Open a Pull Request** with detailed description

---

## 📚 Research & Educational Use

This project is designed for:

- 🎓 **Learning Cryptography** - Understand how block ciphers work internally
- 🔬 **Research** - Experiment with hybrid cipher designs
- 🏫 **Teaching** - Demonstrate cryptographic concepts with working code
- 🧪 **Prototyping** - Quickly test new cipher component combinations

### Related Topics

- **Block Cipher Design** - Feistel networks, SPN structures, key schedules
- **Cryptanalysis** - Understanding cipher weaknesses through composition
- **LLM Applications** - RAG systems for code understanding
- **Automated Testing** - Validation frameworks for security-critical code

---

## 📝 License

This project is for **educational and research purposes only**. Individual algorithm implementations follow their respective specifications and standards.

### Important Security Warning

⚠️ **DO NOT USE IN PRODUCTION**

This implementation is for learning and experimentation only. For production systems:

- Use well-tested libraries: `cryptography`, `PyCrypto`, `OpenSSL`
- Never implement your own cryptography for real-world security
- Consult security professionals for production deployments
- Follow industry best practices (NIST, FIPS)

---

## 🙏 Acknowledgments

### Technologies

- **[DeepSeek AI](https://www.deepseek.com/)** - DeepSeek Coder 7B model
- **[Hugging Face](https://huggingface.co/)** - Model hosting and transformers library
- **[Gradio](https://gradio.app/)** - Beautiful web interface framework
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[FAISS](https://github.com/facebookresearch/faiss)** - Efficient similarity search
- **[Sentence Transformers](https://www.sbert.net/)** - Embeddings library

### Research

- **Cryptographic Community** - Algorithm specifications (NIST, IEEE)
- **AES Competition** - Rijndael, Twofish, Serpent designs
- **Feistel Network** - Classic cipher structure research
- **SPN Designs** - Substitution-permutation network theory

### Inspiration

- **OpenSSL** - Comprehensive crypto library
- **PyCryptodome** - Python cryptography implementation
- **CrypTool** - Educational cryptography software

---

## 📧 Contact & Support

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Pull Requests**: Welcome for improvements and new features

---

## 🌟 Star History

If you find this project helpful, please consider:

- ⭐ Giving it a star on GitHub
- 🔗 Sharing with others interested in cryptography
- 🤝 Contributing improvements or new features
- 📝 Citing in academic work (if applicable)

---

## 📊 Project Statistics

- **Total Lines of Code**: ~10,000+
- **Algorithms Implemented**: 12
- **Cryptographic Components**: 91 functions
- **Test Coverage**: Comprehensive round-trip testing
- **Supported Interfaces**: 2 (RAG Chat + Composer)
- **Data Files**: 8 (embeddings, metadata, datasets)
- **Documentation**: Comprehensive README + inline comments

---

**Made with ❤️ for the cryptography community**

_Version 1.0 - January 2026_

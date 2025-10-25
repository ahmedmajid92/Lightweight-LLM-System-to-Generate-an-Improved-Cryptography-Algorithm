import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from typing import Dict, List, Tuple
import inspect
import ast
import os

def load_algorithm_data():
    """Load algorithm metadata and implementation details."""
    with open('data/algorithms.json', 'r') as f:
        algorithms = json.load(f)
    
    with open('data/algorithm_implementations.json', 'r') as f:
        implementations = json.load(f)
    
    return algorithms, implementations

def extract_function_code(file_path: str, function_name: str) -> str:
    """Extract source code of a specific function from a Python file."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Get the function's source code
                lines = source.split('\n')
                start_line = node.lineno - 1
                
                # Find the end of the function
                end_line = start_line + 1
                indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
                
                for i in range(start_line + 1, len(lines)):
                    line = lines[i]
                    if line.strip() == '':
                        continue
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level and line.strip():
                        break
                    end_line = i + 1
                
                return '\n'.join(lines[start_line:end_line])
    except Exception as e:
        print(f"Error extracting {function_name}: {e}")
        return f"# Function {function_name} implementation not found"
    
    return f"# Function {function_name} not found in {file_path}"

def generate_algorithm_level_chunks(algorithms: List[Dict], implementations: List[Dict]) -> List[Dict]:
    """Generate algorithm-level chunks for cipher recommendation queries."""
    chunks = []
    
    # Create a mapping from algorithm name to implementation
    impl_map = {impl['Algorithm']: impl for impl in implementations}
    
    for algo in algorithms:
        algo_name = algo['Algorithm']
        
        # Combine metadata and implementation info
        text_parts = []
        
        # Algorithm name and year
        text_parts.append(f"Algorithm: {algo_name}")
        if 'year' in algo:
            text_parts.append(f"Year: {algo['year']}")
        
        # Technical specifications
        spec_fields = ['Block Size', 'Key_Size', 'Rounds', 'Structure', 'S-box', 'Math Complexity']
        for field in spec_fields:
            if field in algo:
                text_parts.append(f"{field}: {algo[field]}")
        
        # Performance and security ratings
        perf_fields = ['Speed (1-5)', 'Security (1-5)']
        for field in perf_fields:
            if field in algo:
                text_parts.append(f"{field}: {algo[field]}")
        
        # Usage characteristics
        usage_fields = ['Type', 'Data Size', 'Modes', 'Online/Offline', 'Data Sensitivity']
        for field in usage_fields:
            if field in algo:
                text_parts.append(f"{field}: {algo[field]}")
        
        # Applications and use cases
        if 'Applications' in algo:
            text_parts.append(f"Applications: {algo['Applications']}")
        if 'Data Types' in algo:
            text_parts.append(f"Data Types: {algo['Data Types']}")
        
        # Implementation details from algorithm_implementations.json
        if algo_name in impl_map:
            impl = impl_map[algo_name]
            text_parts.append(f"Block Size: {impl['BlockSize']} bytes")
            text_parts.append(f"Supported Key Sizes: {impl['KeySizes']} bytes")
            if 'Parameters' in impl:
                for param, value in impl['Parameters'].items():
                    text_parts.append(f"{param}: {value}")
        
        # Create chunk
        chunk_text = '; '.join(text_parts)
        chunk_id = f"algo_meta_{algo_name}"
        description = f"Complete algorithm specification for {algo_name} including technical details, performance characteristics, and use cases"
        
        chunks.append({
            'chunk_id': chunk_id,
            'text': chunk_text,
            'description': description,
            'type': 'algorithm_level',
            'algorithm': algo_name
        })
    
    return chunks

def generate_component_level_chunks(implementations: List[Dict]) -> List[Dict]:
    """Generate component-level chunks for specific implementation queries."""
    chunks = []
    
    for impl in implementations:
        algo_name = impl['Algorithm']
        
        # Main encrypt/decrypt functions
        for func_type in ['EncryptFn', 'DecryptFn']:
            if func_type in impl and impl[func_type]:
                func_name = impl[func_type]
                
                # Extract function code
                func_code = extract_function_code(f"src/{impl['File']}", func_name)
                
                # Create descriptive text
                operation = "encryption" if func_type == 'EncryptFn' else "decryption"
                text_parts = [
                    f"Algorithm: {algo_name}",
                    f"Function: {func_name}",
                    f"Operation: {operation}",
                    f"Block size: {impl['BlockSize']} bytes",
                    f"Implementation:\n{func_code}"
                ]
                
                chunk_text = '\n'.join(text_parts)
                chunk_id = f"{algo_name}.{func_name}"
                description = f"{algo_name} {operation} function implementation"
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'description': description,
                    'type': 'component_level',
                    'algorithm': algo_name,
                    'component': func_name,
                    'operation': operation
                })
        
        # Component functions
        if 'ComponentFns' in impl:
            for component_fn in impl['ComponentFns']:
                # Try both Components.py and AlgorithmsBlock.py
                func_code = extract_function_code(f"src/{impl['ComponentFile']}", component_fn)
                if "not found" in func_code:
                    func_code = extract_function_code(f"src/{impl['File']}", component_fn)
                
                # Create descriptive text for component
                text_parts = [
                    f"Algorithm: {algo_name}",
                    f"Component: {component_fn}",
                    f"Type: Helper function",
                    f"Implementation:\n{func_code}"
                ]
                
                # Add context based on function name
                if 'sbox' in component_fn.lower() or 'sub_bytes' in component_fn:
                    text_parts.insert(-1, "Purpose: Substitution layer (S-box operations)")
                elif 'shift' in component_fn.lower() or 'permut' in component_fn.lower():
                    text_parts.insert(-1, "Purpose: Permutation layer")
                elif 'mix' in component_fn.lower():
                    text_parts.insert(-1, "Purpose: Linear diffusion layer")
                elif 'key' in component_fn.lower():
                    text_parts.insert(-1, "Purpose: Key schedule operations")
                elif 'round' in component_fn.lower():
                    text_parts.insert(-1, "Purpose: Round function")
                
                chunk_text = '\n'.join(text_parts)
                chunk_id = f"{algo_name}.{component_fn}"
                description = f"{algo_name} {component_fn} component implementation"
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'description': description,
                    'type': 'component_level',
                    'algorithm': algo_name,
                    'component': component_fn
                })
        
        # Key schedule function (if separate)
        if 'KeyScheduleFn' in impl and impl['KeyScheduleFn']:
            func_name = impl['KeyScheduleFn']
            func_code = extract_function_code(f"src/{impl['ComponentFile']}", func_name)
            
            text_parts = [
                f"Algorithm: {algo_name}",
                f"Component: {func_name}",
                f"Type: Key schedule function",
                f"Purpose: Generate round keys from master key",
                f"Implementation:\n{func_code}"
            ]
            
            chunk_text = '\n'.join(text_parts)
            chunk_id = f"{algo_name}.{func_name}"
            description = f"{algo_name} key schedule implementation"
            
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'description': description,
                'type': 'component_level',
                'algorithm': algo_name,
                'component': func_name,
                'operation': 'key_schedule'
            })
    
    return chunks

def generate_embeddings(chunks: List[Dict]) -> Dict:
    """Generate embeddings for all chunks using a reliable model."""
    
    # Try different embedding models in order of preference
    models_to_try = [
        'all-MiniLM-L6-v2',  # Reliable general model
        'sentence-transformers/all-MiniLM-L6-v2',  # Full path
        'microsoft/codebert-base',  # Code-specific alternative
        'huggingface/CodeBERTa-small-v1'  # Original choice
    ]
    
    embed_model = None
    model_name = None
    
    for model in models_to_try:
        try:
            print(f"Trying to load model: {model}")
            embed_model = SentenceTransformer(model)
            model_name = model
            print(f"✅ Successfully loaded: {model}")
            break
        except Exception as e:
            print(f"❌ Failed to load {model}: {e}")
            continue
    
    if embed_model is None:
        raise Exception("Could not load any embedding model!")
    
    # Extract texts for embedding
    texts = [chunk['text'] for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create embedding database
    embedding_db = {
        'chunks': chunks,
        'embeddings': embeddings,
        'model_name': model_name,
        'chunk_index': {chunk['chunk_id']: i for i, chunk in enumerate(chunks)}
    }
    
    return embedding_db

def main():
    """Generate and save embeddings for the algorithm database."""
    print("Loading algorithm data...")
    algorithms, implementations = load_algorithm_data()
    
    print("Generating algorithm-level chunks...")
    algo_chunks = generate_algorithm_level_chunks(algorithms, implementations)
    print(f"Created {len(algo_chunks)} algorithm-level chunks")
    
    print("Generating component-level chunks...")
    comp_chunks = generate_component_level_chunks(implementations)
    print(f"Created {len(comp_chunks)} component-level chunks")
    
    # Combine all chunks
    all_chunks = algo_chunks + comp_chunks
    print(f"Total chunks: {len(all_chunks)}")
    
    # Generate embeddings
    embedding_db = generate_embeddings(all_chunks)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to disk
    with open('data/algorithm_embeddings.pkl', 'wb') as f:
        pickle.dump(embedding_db, f)
    
    print(f"\n✅ Successfully saved embeddings using model: {embedding_db['model_name']}")
    print("📁 Saved to: data/algorithm_embeddings.pkl")
    
    # Test a few queries
    print("\n🧪 Testing embeddings...")
    
    # Load the same model that was used for embeddings
    embed_model = SentenceTransformer(embedding_db['model_name'])
    
    test_queries = [
        "mix columns",
        "blowfish full implementation", 
        "AES key expansion"
    ]
    
    for query in test_queries:
        query_emb = embed_model.encode([query])
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(embedding_db['embeddings'], query_emb.T).flatten()
        top_idx = np.argmax(similarities)
        
        best_chunk = embedding_db['chunks'][top_idx]
        print(f"\n📝 Query: '{query}'")
        print(f"🏆 Best match: {best_chunk['chunk_id']} (similarity: {similarities[top_idx]:.3f})")

if __name__ == "__main__":
    main()
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Tuple
import importlib.util
import inspect
import textwrap
import pathlib
import re
import os

class EnhancedRAGChat:
    def __init__(self, 
                 embedding_file: str = 'data/algorithm_embeddings.pkl',
                 impl_file: str = 'data/algorithm_implementations.json',
                 model_name: str = 'deepseek-ai/deepseek-coder-7b-instruct-v1.5'):
        
        print("Loading embedding database...")
        with open(embedding_file, 'rb') as f:
            self.embedding_db = pickle.load(f)
        
        print("Loading algorithm implementations...")
        with open(impl_file, 'r') as f:
            self.implementations = {impl['Algorithm']: impl for impl in json.load(f)}
        
        print("Loading CodeBERTa embedding model...")
        # Use the same model that was used to create embeddings
        embed_model_name = self.embedding_db.get('model_name', 'huggingface/CodeBERTa-small-v1')
        self.embed_model = SentenceTransformer(embed_model_name)
        print(f"Using embedding model: {embed_model_name}")
        
        print(f"Loading DeepSeek Coder model: {model_name}...")
        self._load_llm(model_name)
        
        print("RAG system ready!")
    
    def _load_llm(self, model_name: str):
        """Load the DeepSeek Coder language model."""
        # Configure quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            use_fast=False
        )
        
        # Load model with quantization
        print("Loading model (this may take a few minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Set pad token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_code_snippet(self, file_path: str, fn_name: str) -> str:
        """Extract source code for a specific function."""
        try:
            # Handle relative paths
            if not os.path.isabs(file_path):
                file_path = os.path.join('src', file_path)
            
            if not os.path.exists(file_path):
                return f"# File {file_path} not found"
            
            module_name = pathlib.Path(file_path).stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return f"# Could not load module from {file_path}"
            
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            
            if not hasattr(m, fn_name):
                return f"# Function {fn_name} not found in {file_path}"
            
            func = getattr(m, fn_name)
            source = inspect.getsource(func)
            return textwrap.dedent(source)
            
        except Exception as e:
            return f"# Error extracting {fn_name} from {file_path}: {str(e)}"
    
    def retrieve_chunks(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Dict]:
        """Retrieve most relevant chunks for a query."""
        # Generate query embedding
        query_emb = self.embed_model.encode([query])
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(self.embedding_db['embeddings'], query_emb.T).flatten()
        
        # Get top-k most similar chunks above threshold
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= min_similarity:
                chunk = self.embedding_db['chunks'][idx].copy()
                chunk['similarity'] = float(similarities[idx])
                results.append(chunk)
        
        return results
    
    def classify_query_intent(self, query: str) -> str:
        """Classify query intent to determine retrieval strategy."""
        query_lower = query.lower()
        
        # 1) Anything explicitly asking for "component(s)" → code intent (highest priority)
        if re.search(r"\b(component|helper|function)s?\b", query_lower):
            return 'code'
        
        # 2) Code/implementation-specific queries
        code_pattern = r"\b(code|implementation|encrypt_block|decrypt_block|source|show\s+me.*implementation|give\s+me.*code|full.*implementation|complete.*implementation)\b"
        if re.search(code_pattern, query_lower):
            return 'code'
        
        # 3) Algorithm recommendation queries (more specific phrases)
        rec_keywords = [
            'recommend', 'suggest', 'choose', 'best', 'which cipher',
            'for high security', 'for performance', 'for speed', 'for application',
            'for data protection', 'for encryption', 'security level', 'fast cipher'
        ]
        if any(keyword in query_lower for keyword in rec_keywords):
            return 'recommendation'
        
        # 4) General information queries
        return 'general'
    
    def adaptive_retrieve(self, query: str, intent: str) -> List[Dict]:
        """Retrieve chunks based on query intent."""
        if intent == 'code':
            # For code queries, retrieve component-level chunks with lower threshold
            component_chunks = self.retrieve_chunks(query, top_k=10, min_similarity=0.2)
            component_chunks = [c for c in component_chunks if c['type'] == 'component_level']
            
            if component_chunks:
                return component_chunks[:5]  # Return top 5 component chunks
            else:
                # Fallback to algorithm-level if no components found
                return self.retrieve_chunks(query, top_k=3, min_similarity=0.2)
        
        elif intent == 'recommendation':
            # Prioritize algorithm-level chunks
            algo_chunks = self.retrieve_chunks(query, top_k=5, min_similarity=0.3)
            algo_chunks = [c for c in algo_chunks if c['type'] == 'algorithm_level']
            return algo_chunks[:3]  # Return top 3 algorithms
        
        else:  # general
            # Mixed retrieval
            return self.retrieve_chunks(query, top_k=4, min_similarity=0.3)
    
    def assemble_full_algorithm_code(self, algorithm: str) -> str:
        """Assemble complete algorithm implementation including all components."""
        if algorithm not in self.implementations:
            return f"# Implementation details for {algorithm} not found"
        
        impl = self.implementations[algorithm]
        code_parts = [f"# ===== {algorithm} Complete Implementation =====\n"]
        
        # Add main encrypt/decrypt functions
        if impl.get('EncryptFn'):
            code_parts.append(f"# Main encryption function:")
            code_parts.append(self.get_code_snippet(impl['File'], impl['EncryptFn']))
            code_parts.append("")
        
        if impl.get('DecryptFn'):
            code_parts.append(f"# Main decryption function:")
            code_parts.append(self.get_code_snippet(impl['File'], impl['DecryptFn']))
            code_parts.append("")
        
        # Add key schedule function if separate
        if impl.get('KeyScheduleFn'):
            code_parts.append(f"# Key schedule function:")
            code_parts.append(self.get_code_snippet(impl.get('ComponentFile', impl['File']), impl['KeyScheduleFn']))
            code_parts.append("")
        
        # Add all component functions
        if impl.get('ComponentFns'):
            code_parts.append(f"# Component functions:")
            for component_fn in impl['ComponentFns']:
                code_parts.append(f"# --- {component_fn} ---")
                # Try component file first, then main file
                component_file = impl.get('ComponentFile', impl['File'])
                code_parts.append(self.get_code_snippet(component_file, component_fn))
                code_parts.append("")
        
        return "\n".join(code_parts)
    
    def assemble_components_only(self, chunks: List[Dict]) -> str:
        """Assemble only the component functions from retrieved chunks."""
        code_parts = ["# ===== Component Functions =====\n"]
        
        for chunk in chunks:
            if chunk['type'] == 'component_level' and 'component' in chunk:
                algorithm = chunk['algorithm']
                component = chunk['component']
                
                if algorithm in self.implementations:
                    impl = self.implementations[algorithm]
                    component_file = impl.get('ComponentFile', impl['File'])
                    
                    code_parts.append(f"# --- {algorithm}: {component} ---")
                    code_parts.append(f"# Similarity: {chunk['similarity']:.3f}")
                    code_parts.append(self.get_code_snippet(component_file, component))
                    code_parts.append("")
        
        return "\n".join(code_parts)
    
    def handle_code_query(self, query: str, chunks: List[Dict]) -> str:
        """Handle code-specific queries without using LLM."""
        query_lower = query.lower()
        
        # 1. First, check for specific component name requests
        # This gives priority to exact component matches
        for impl in self.implementations.values():
            if impl.get('ComponentFns'):
                for component_fn in impl['ComponentFns']:
                    # Check if the component name is mentioned in the query
                    component_variants = [
                        component_fn.lower(),
                        component_fn.replace('_', ' ').lower(),  # e.g., "mix_columns" -> "mix columns"
                        component_fn.replace('_', '').lower(),   # e.g., "mix_columns" -> "mixcolumns"
                    ]
                    
                    if any(variant in query_lower for variant in component_variants):
                        algorithm = impl['Algorithm']
                        code_parts = [f"# ===== {algorithm}: {component_fn} =====\n"]
                        
                        # Determine which file contains the function
                        if component_fn == impl.get('EncryptFn') or component_fn == impl.get('DecryptFn'):
                            file_path = impl['File']
                        else:
                            file_path = impl.get('ComponentFile', impl['File'])
                        
                        code_parts.append(f"# Component: {component_fn}")
                        code_parts.append(f"# Algorithm: {algorithm}")
                        code_parts.append(f"# File: {file_path}")
                        code_parts.append("")
                        code_parts.append(self.get_code_snippet(file_path, component_fn))
                        
                        return "\n".join(code_parts)
    
        # Also check main encrypt/decrypt functions
        for impl in self.implementations.values():
            for func_type in ['EncryptFn', 'DecryptFn']:
                if impl.get(func_type):
                    func_name = impl[func_type]
                    func_variants = [
                        func_name.lower(),
                        func_name.replace('_', ' ').lower(),
                        func_name.replace('_', '').lower(),
                    ]
                    
                    if any(variant in query_lower for variant in func_variants):
                        algorithm = impl['Algorithm']
                        operation = "encryption" if func_type == 'EncryptFn' else "decryption"
                        
                        code_parts = [f"# ===== {algorithm}: {func_name} =====\n"]
                        code_parts.append(f"# Function: {func_name}")
                        code_parts.append(f"# Operation: {operation}")
                        code_parts.append(f"# Algorithm: {algorithm}")
                        code_parts.append("")
                        code_parts.append(self.get_code_snippet(impl['File'], func_name))
                        
                        return "\n".join(code_parts)
    
        # 2. Check for full implementation requests (improved regex)
        full_impl_pattern = r"\b(full|complete|entire|whole)\b.*\b(implementation|algorithm|code)\b"
        if re.search(full_impl_pattern, query_lower):
            # Try to extract algorithm name from the query
            for alg in self.implementations:
                if alg.lower() in query_lower:
                    return self.assemble_full_algorithm_code(alg)
            
            # Fallback to first chunk's algorithm if no explicit name found
            if chunks:
                return self.assemble_full_algorithm_code(chunks[0]['algorithm'])
    
        # 3. Check for components-only requests
        components_only_pattern = r"\b(component|helper|function)s?\s+only\b"
        if re.search(components_only_pattern, query_lower):
            # 3a) Find algorithm name in the query
            alg = next((a for a in self.implementations if a.lower() in query_lower), None)
            if alg:
                # 3b) Keep only that algorithm's chunks
                alg_chunks = [c for c in chunks if c['algorithm'].lower() == alg.lower()]
                if alg_chunks:
                    return self.assemble_components_only(alg_chunks)
                else:
                    # If no chunks found for that algorithm, generate directly from implementation
                    return self._assemble_algorithm_components_direct(alg)
            # Fallback if we couldn't detect the algorithm name
            return self.assemble_components_only(chunks)
    
        # 4. Check for algorithm-specific requests (e.g., "give me Blowfish code")
        for alg in self.implementations:
            if alg.lower() in query_lower:
                # If they want a specific algorithm, prioritize its components
                alg_chunks = [c for c in chunks if c.get('algorithm', '').lower() == alg.lower()]
                if alg_chunks:
                    chunks = alg_chunks[:3] + [c for c in chunks if c not in alg_chunks][:2]
                    break
    
        # 5. Default: Return the most relevant code snippets
        code_parts = ["# ===== Relevant Code Snippets =====\n"]
        
        for i, chunk in enumerate(chunks[:3], 1):  # Top 3 matches
            if chunk['type'] == 'component_level':
                algorithm = chunk['algorithm']
                component = chunk.get('component', 'unknown')
                
                code_parts.append(f"# {i}. {algorithm}: {component}")
                code_parts.append(f"# Similarity: {chunk['similarity']:.3f}")
                code_parts.append(f"# Description: {chunk['description']}")
                
                if algorithm in self.implementations:
                    impl = self.implementations[algorithm]
                    
                    # Determine which file contains the function
                    if component == impl.get('EncryptFn') or component == impl.get('DecryptFn'):
                        file_path = impl['File']
                    else:
                        file_path = impl.get('ComponentFile', impl['File'])
                
                    code_parts.append(self.get_code_snippet(file_path, component))
                
                code_parts.append("")
        
        return "\n".join(code_parts)
    
    def _assemble_algorithm_components_direct(self, algorithm: str) -> str:
        """Assemble components for a specific algorithm directly from implementation data."""
        if algorithm not in self.implementations:
            return f"# Components for {algorithm} not found"
        
        impl = self.implementations[algorithm]
        code_parts = [f"# ===== {algorithm} Component Functions =====\n"]
        
        # Add all component functions
        if impl.get('ComponentFns'):
            for component_fn in impl['ComponentFns']:
                code_parts.append(f"# --- {component_fn} ---")
                # Try component file first, then main file
                component_file = impl.get('ComponentFile', impl['File'])
                code_parts.append(self.get_code_snippet(component_file, component_fn))
                code_parts.append("")
        
        # Add key schedule function if separate
        if impl.get('KeyScheduleFn'):
            code_parts.append(f"# --- {impl['KeyScheduleFn']} (Key Schedule) ---")
            code_parts.append(self.get_code_snippet(impl.get('ComponentFile', impl['File']), impl['KeyScheduleFn']))
            code_parts.append("")
        
        if len(code_parts) == 1:  # Only the header was added
            code_parts.append(f"# No component functions found for {algorithm}")
        
        return "\n".join(code_parts)
    
    def format_context(self, chunks: List[Dict], intent: str) -> str:
        """Format retrieved chunks for LLM context."""
        if not chunks:
            return "No relevant information found."
        
        context_parts = []
        
        if intent == 'recommendation':
            context_parts.append("=== ALGORITHM OPTIONS ===")
            for i, chunk in enumerate(chunks, 1):
                context_parts.append(f"\n{i}. {chunk['algorithm']}")
                context_parts.append(f"Match Score: {chunk['similarity']:.3f}")
                # Extract key characteristics
                text = chunk['text']
                key_info = []
                for line in text.split(';'):
                    line = line.strip()
                    if any(keyword in line.lower() for keyword in 
                           ['security', 'speed', 'rounds', 'structure', 'applications']):
                        key_info.append(line)
                context_parts.append('\n'.join(key_info[:5]))  # Top 5 characteristics
                context_parts.append("")
        
        else:  # general
            context_parts.append("=== RELEVANT INFORMATION ===")
            for i, chunk in enumerate(chunks, 1):
                context_parts.append(f"\n{i}. {chunk['description']}")
                context_parts.append(f"Type: {chunk['type'].replace('_', ' ').title()}")
                context_parts.append(f"Relevance: {chunk['similarity']:.3f}")
                # Show first few lines
                text_lines = chunk['text'].split('\n')[:10]
                context_parts.append('\n'.join(text_lines))
                context_parts.append("")
        
        return '\n'.join(context_parts)
    
    def generate_response(self, query: str, context: str, intent: str) -> str:
        """Generate response using DeepSeek Coder with retrieved context."""
        
        if intent == 'recommendation':
            system_prompt = (
                "You are a cryptography consultant. Analyze the user's requirements "
                "and recommend the most suitable cipher based on the provided specifications."
            )
        else:
            system_prompt = (
                "You are a cryptography expert. Answer the user's question using "
                "the provided context about cryptographic algorithms."
            )
        
        # DeepSeek Coder uses a specific chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=4000
        ).to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "assistant" in response:
            # Split by the last assistant marker to get only the new response
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        return response
    
    def chat(self, query: str) -> Dict:
        """Main chat interface - process query and return response with metadata."""
        
        # Classify query intent
        intent = self.classify_query_intent(query)
        
        # Retrieve relevant chunks
        chunks = self.adaptive_retrieve(query, intent)
        
        if intent == 'code':
            # Handle code queries directly without LLM
            response = self.handle_code_query(query, chunks)
            
            return {
                'query': query,
                'intent': intent,
                'response': response,
                'retrieved_chunks': len(chunks),
                'chunk_details': [
                    {
                        'id': chunk['chunk_id'],
                        'type': chunk['type'],
                        'algorithm': chunk['algorithm'],
                        'similarity': chunk['similarity']
                    } for chunk in chunks
                ],
                'method': 'direct_code_retrieval'
            }
        
        else:
            # Use LLM for recommendation and general queries
            context = self.format_context(chunks, intent)
            response = self.generate_response(query, context, intent)
            
            return {
                'query': query,
                'intent': intent,
                'response': response,
                'retrieved_chunks': len(chunks),
                'chunk_details': [
                    {
                        'id': chunk['chunk_id'],
                        'type': chunk['type'],
                        'algorithm': chunk['algorithm'],
                        'similarity': chunk['similarity']
                    } for chunk in chunks
                ],
                'method': 'llm_generation'
            }

def main():
    """Interactive chat interface."""
    try:
        rag = EnhancedRAGChat()
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return
    
    print("\n🔐 Enhanced Cryptography RAG Chat")
    print("Ask me about algorithms, implementations, or recommendations!")
    print("Examples:")
    print("  - 'Show me AES mix columns implementation'")
    print("  - 'Give me full Blowfish implementation'")
    print("  - 'Recommend a cipher for high security'")
    print("  - 'Components only for DES'")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            if not query or query.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            print("\n🤔 Processing...")
            result = rag.chat(query)
            
            print(f"\n🧠 Intent: {result['intent']}")
            print(f"🔧 Method: {result['method']}")
            print(f"📊 Retrieved {result['retrieved_chunks']} relevant chunks")
            
            if result['intent'] == 'code':
                print(f"\n💻 Code Response:")
                print("```python")
                print(result['response'])
                print("```")
            else:
                print(f"\n🤖 Assistant: {result['response']}")
            
            # Show sources
            if result['chunk_details']:
                print(f"\n📚 Sources:")
                for chunk in result['chunk_details'][:3]:  # Show top 3
                    print(f"  - {chunk['id']} ({chunk['type']}, similarity: {chunk['similarity']:.3f})")
            
            print("\n" + "─"*50 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
"""
Script to clean up old Llama model files and update to DeepSeek Coder.
"""

import os
import shutil
from pathlib import Path

def cleanup_old_models():
    """Remove old Llama model files and directories."""
    
    old_paths = [
        'data/Llama-3.2-3B-Instruct',
        'models/llama-3b-lora',
        # Add any other Llama-related paths you want to remove
    ]
    
    print("🧹 Cleaning up old model files...")
    
    for path in old_paths:
        full_path = Path(path)
        if full_path.exists():
            if full_path.is_dir():
                print(f"📁 Removing directory: {path}")
                try:
                    shutil.rmtree(full_path)
                    print(f"✅ Successfully removed {path}")
                except Exception as e:
                    print(f"❌ Error removing {path}: {e}")
            elif full_path.is_file():
                print(f"📄 Removing file: {path}")
                try:
                    full_path.unlink()
                    print(f"✅ Successfully removed {path}")
                except Exception as e:
                    print(f"❌ Error removing {path}: {e}")
        else:
            print(f"⚠️  Path not found: {path}")
    
    print("\n🎉 Cleanup complete!")
    print("💡 The new system will download DeepSeek Coder automatically when first run.")

def check_disk_space():
    """Check available disk space before cleanup."""
    try:
        import psutil
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        print(f"💾 Available disk space: {free_gb:.1f} GB")
        
        if free_gb < 10:
            print("⚠️  Warning: Low disk space. DeepSeek Coder 7B requires ~13-15 GB.")
            return False
        return True
    except ImportError:
        print("💡 Install psutil to check disk space: pip install psutil")
        return True

def main():
    print("🔄 Model Migration: Llama → DeepSeek Coder")
    print("=" * 50)
    
    # Check disk space
    if not check_disk_space():
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Confirm cleanup
    print("\nThis will remove:")
    print("- data/Llama-3.2-3B-Instruct/")
    print("- models/llama-3b-lora/")
    print("- Any associated cache files")
    
    response = input("\nProceed with cleanup? (y/N): ")
    if response.lower() == 'y':
        cleanup_old_models()
        
        print("\n🚀 Next steps:")
        print("1. Run: python src/enhanced_rag_chat.py")
        print("2. The system will automatically download DeepSeek Coder (~13-15 GB)")
        print("3. First run may take 10-15 minutes for model download")
        
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    main()
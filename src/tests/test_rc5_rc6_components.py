"""
Test RC5 and RC6 component functions
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.Components import rc5_key_schedule, rc6_key_schedule, rotate_left

def test_rc5_key_schedule():
    """Test RC5 key schedule generation."""
    print("Testing RC5 Key Schedule...")
    
    # Test with a simple key
    key = b"TestKey123456"  # 13 bytes
    w = 32  # word size
    r = 12  # rounds
    
    # Generate S-array
    S = rc5_key_schedule(key, w, r)
    
    # Verify S-array size
    expected_size = 2 * (r + 1)  # = 26
    assert len(S) == expected_size, f"Expected {expected_size} words, got {len(S)}"
    
    # Verify S-array contains valid 32-bit words
    for i, word in enumerate(S):
        assert 0 <= word < (1 << 32), f"S[{i}] = {word} is not a valid 32-bit word"
    
    print(f"  ✅ Generated S-array with {len(S)} words")
    print(f"  ✅ First few words: {[hex(S[i]) for i in range(min(4, len(S)))]}")
    print(f"  ✅ RC5 key schedule works correctly!")
    return S

def test_rc6_key_schedule():
    """Test RC6 key schedule generation."""
    print("\nTesting RC6 Key Schedule...")
    
    # Test with a simple key
    key = b"TestKey123456789"  # 16 bytes
    w = 32  # word size
    r = 20  # rounds
    
    # Generate S-array
    S = rc6_key_schedule(key, w, r)
    
    # Verify S-array size
    expected_size = 2 * r + 4  # = 44
    assert len(S) == expected_size, f"Expected {expected_size} words, got {len(S)}"
    
    # Verify S-array contains valid 32-bit words
    for i, word in enumerate(S):
        assert 0 <= word < (1 << 32), f"S[{i}] = {word} is not a valid 32-bit word"
    
    print(f"  ✅ Generated S-array with {len(S)} words")
    print(f"  ✅ First few words: {[hex(S[i]) for i in range(min(4, len(S)))]}")
    print(f"  ✅ RC6 key schedule works correctly!")
    return S

def test_rc5_with_encryption():
    """Test RC5 key schedule with actual encryption."""
    print("\nTesting RC5 with Encryption...")
    
    from src.AlgorithmsBlock import rc5_encrypt_block, rc5_decrypt_block
    from src.Components import bytes_to_int, int_to_bytes
    
    key = b"SecretKey1234567"  # 16 bytes
    plaintext = b"TestData"  # 8 bytes
    
    # Generate key schedule
    S = rc5_key_schedule(key, w=32, r=12)
    
    # Encrypt
    pt_int = bytes_to_int(plaintext)
    ct_int = rc5_encrypt_block(pt_int, S, w=32, r=12)
    
    # Decrypt
    decrypted_int = rc5_decrypt_block(ct_int, S, w=32, r=12)
    decrypted = int_to_bytes(decrypted_int, 8)
    
    print(f"  Plaintext:  {plaintext}")
    print(f"  Ciphertext: {hex(ct_int)}")
    print(f"  Decrypted:  {decrypted}")
    
    assert decrypted == plaintext, "Decryption failed!"
    print(f"  ✅ RC5 encryption/decryption cycle successful!")

def test_rc6_with_encryption():
    """Test RC6 key schedule with actual encryption."""
    print("\nTesting RC6 with Encryption...")
    
    from src.AlgorithmsBlock import rc6_encrypt_block, rc6_decrypt_block
    from src.Components import bytes_to_int, int_to_bytes
    
    key = b"SecretKey1234567"  # 16 bytes
    plaintext = b"TestData12345678"  # 16 bytes
    
    # Generate key schedule
    S = rc6_key_schedule(key, w=32, r=20)
    
    # Encrypt
    pt_int = bytes_to_int(plaintext)
    ct_int = rc6_encrypt_block(pt_int, S, w=32, r=20)
    
    # Decrypt
    decrypted_int = rc6_decrypt_block(ct_int, S, w=32, r=20)
    decrypted = int_to_bytes(decrypted_int, 16)
    
    print(f"  Plaintext:  {plaintext}")
    print(f"  Ciphertext: {hex(ct_int)}")
    print(f"  Decrypted:  {decrypted}")
    
    assert decrypted == plaintext, "Decryption failed!"
    print(f"  ✅ RC6 encryption/decryption cycle successful!")

def test_component_registry():
    """Test that component registry detects RC5/RC6 functions."""
    print("\nTesting Component Registry Detection...")
    
    from src.component_registry import scan_components, list_algorithms
    
    # Get all detected algorithms
    algorithms = list_algorithms()
    
    print(f"  Detected algorithms: {algorithms}")
    
    # Check if RC5 and RC6 are detected
    assert "RC5" in algorithms, "RC5 not detected in component registry!"
    assert "RC6" in algorithms, "RC6 not detected in component registry!"
    
    # Get RC5 components
    from src.component_registry import list_components_for_algorithm
    rc5_components = list_components_for_algorithm("RC5")
    rc6_components = list_components_for_algorithm("RC6")
    
    print(f"  RC5 components: {[c.name for c in rc5_components]}")
    print(f"  RC6 components: {[c.name for c in rc6_components]}")
    
    # Verify key schedule functions are detected
    rc5_funcs = [c.name for c in rc5_components]
    rc6_funcs = [c.name for c in rc6_components]
    
    assert "rc5_key_schedule" in rc5_funcs, "rc5_key_schedule not detected!"
    assert "rc6_key_schedule" in rc6_funcs, "rc6_key_schedule not detected!"
    
    print(f"  ✅ Component registry correctly detects RC5/RC6 functions!")

if __name__ == "__main__":
    print("="*60)
    print("RC5 & RC6 Component Test Suite")
    print("="*60)
    
    try:
        # Test key schedules
        test_rc5_key_schedule()
        test_rc6_key_schedule()
        
        # Test with actual encryption/decryption
        test_rc5_with_encryption()
        test_rc6_with_encryption()
        
        # Test component registry
        test_component_registry()
        
        print("\n" + "="*60)
        print("✅ All RC5/RC6 tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


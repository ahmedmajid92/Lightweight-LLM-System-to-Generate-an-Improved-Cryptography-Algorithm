from Components import *


# ========================
# Encryption Blocks
# ========================

def aes_encrypt_block(plaintext_bytes: bytes, master_key: bytes) -> bytes:
    # 1) Key-expansion → round_keys: List[List[int]] 
    round_keys = aes_key_expansion(master_key)

    # 2) State ← plaintext as 16-byte list
    state = list(plaintext_bytes)

    # 3) Initial AddRoundKey
    state = add_round_key(state, round_keys[0])

    # 4) For rounds 1…Nr–1:
    for rk in round_keys[1:-1]:
        state = sub_bytes(state)                # AES S-box layer
        state = shift_rows(state)               # row-wise permutation
        state = mix_columns(state)              # GF(2^8) linear layer
        state = add_round_key(state, rk)        # round key mixing

    # 5) Final round (no MixColumns)
    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, round_keys[-1])

    # 6) Serialize back to bytes
    return bytes(state)

def des_encrypt_block(block64: int, master_key64: int) -> int:
    # 1) Generate 16 × 48-bit subkeys
    subkeys = des_key_schedule(master_key64)

    # 2) Initial Permutation
    state = initial_permutation(block64)

    # 3) 16 Feistel rounds:
    state = feistel_cipher(state, subkeys, des_f, half_size=32)

    # 4) Final Permutation
    return final_permutation(state)


def triple_des_encrypt_block(block64: int, keys: Tuple[int,int,int]) -> int:
    k1,k2,k3 = keys
    c1 = des_encrypt_block(block64, k1)
    c2 = des_decrypt_block(c1,        k2)
    return des_encrypt_block(c2,      k3)


def blowfish_encrypt_block(block64: int, key_bytes: bytes, p_array: List[int], s_boxes: List[List[int]]) -> int:
    # 1) P-array & S-boxes ← blowfish_key_schedule(key_bytes)
    # 2) Split the 64-bit block into two 32-bit halves
    left = (block64 >> 32) & 0xFFFFFFFF
    right = block64 & 0xFFFFFFFF
    
    # 3) Apply the 16 rounds of the Feistel network
    for i in range(16):
        left ^= p_array[i]
        right ^= blowfish_f(left, s_boxes)
        left, right = right, left
    
    # 4) Undo the last swap and apply the final XORs
    left, right = right, left
    right ^= p_array[16]
    left ^= p_array[17]
    
    # 5) Combine the halves back into a 64-bit block
    return (left << 32) | right


def twofish_encrypt_block(block128: int, key_bytes: bytes) -> int:
    """
    Encrypt a 128-bit block using Twofish algorithm.
    """
    # Generate key schedule
    subkeys, q_boxes, mds = twofish_key_schedule(key_bytes)
    
    # Split block into four 32-bit words
    words = []
    for i in range(4):
        words.append((block128 >> (32 * (3-i))) & 0xFFFFFFFF)
    
    # Initial whitening
    words[0] ^= subkeys[0]
    words[1] ^= subkeys[1]
    words[2] ^= subkeys[2]
    words[3] ^= subkeys[3]
    
    # Main encryption rounds
    for i in range(8):  # 16 rounds in pairs
        # Round 2i
        t0 = twofish_f(words[0], q_boxes, mds, subkeys[2*i+4])
        t1 = twofish_f(rotate_left(words[1], 8, 32), q_boxes, mds, subkeys[2*i+5])
        words[2] = rotate_right(words[2] ^ (t0 + t1 + subkeys[2*i+6]), 1, 32)
        words[3] = rotate_left(words[3], 1, 32) ^ (t0 + 2*t1 + subkeys[2*i+7])
        
        # Round 2i+1
        words = [words[2], words[3], words[0], words[1]]  # Swap words
        
    # Final swap and whitening
    words = [words[2], words[3], words[0], words[1]]
    words[0] ^= subkeys[4+2*8]
    words[1] ^= subkeys[5+2*8]
    words[2] ^= subkeys[6+2*8]
    words[3] ^= subkeys[7+2*8]
    
    # Recombine the words
    result = 0
    for i in range(4):
        result = (result << 32) | words[i]
    
    return result

# Add this helper function if needed
def rotate_right(val: int, r_bits: int, width: int) -> int:
    """Rotate val right by r_bits over width bits."""
    r_bits = r_bits % width
    if r_bits < 0:
        r_bits = width + r_bits
    return ((val >> r_bits) | (val << (width - r_bits))) & ((1 << width) - 1)

def serpent_encrypt_block(block128: int, key_bytes: bytes) -> int:
    # Generate round keys
    round_keys = serpent_key_schedule(key_bytes)
    
    state = block128
    
    # Apply 32 rounds
    for r in range(32):
        # Key mixing
        state ^= round_keys[r]
        
        # Apply S-box layer (r mod 8)
        state = apply_sbox_layer(state, [serpent_sboxes[r % 8]], 128, 4)
        
        # Linear transformation (except in the last round)
        if r < 31:
            state = apply_permutation(state, serpent_linear_p, 128)
    
    # Final key mixing
    state ^= round_keys[32]
    
    return state


def camellia_encrypt_block(block128: int, round_keys: List[int], fl_keys: List[int]) -> int:
    """Encrypt a 128-bit block using Camellia algorithm."""
    state = block128 ^ round_keys[0]   # pre-whitening
    
    # 18 Feistel rounds, with FL/FL−1 after rounds 6 and 12:
    for i in range(1, 19):
        state = feistel_rounds_step(state, round_keys[i], camellia_f)
        if i in (6, 12):
            state = camellia_fl(state, fl_keys[i//6 - 1])
    
    # post-whitening
    return state ^ round_keys[-1]


def cast128_encrypt_block(block64: int, round_keys: List[int]) -> int:
    """Encrypt a 64-bit block using CAST-128 algorithm."""
    # Split into two 32-bit halves
    left = (block64 >> 32) & 0xFFFFFFFF
    right = block64 & 0xFFFFFFFF
    
    # Apply 16 rounds
    for i in range(16):
        # In rounds 13-16, the round function is slightly different
        if i >= 12:
            # Subtraction instead of XOR for the last 4 rounds
            right = (right - cast128_f(left, round_keys[i])) & 0xFFFFFFFF
        else:
            right ^= cast128_f(left, round_keys[i])
        
        # Swap left and right except for the last round
        if i < 15:
            left, right = right, left
    
    # Combine the halves
    return (left << 32) | right


def idea_encrypt_block(block64: int, round_keys: List[int]) -> int:
    """Encrypt a 64-bit block using IDEA algorithm."""
    # Split into four 16-bit blocks
    x1, x2, x3, x4 = split_into_four_16bit(block64)
    
    # Apply 8 rounds
    for round in range(8):
        # Extract 6 subkeys for this round
        k = round_keys[round*6:(round+1)*6]
        # Apply round function
        x1, x2, x3, x4 = idea_round(x1, x2, x3, x4, k)
    
    # Final output transformation with last 4 subkeys
    x1, x2, x3, x4 = idea_output_transform(x1, x2, x3, x4, round_keys[-4:])
    
    # Combine into 64-bit result
    return (x1 << 48) | (x2 << 32) | (x3 << 16) | x4


def seed_encrypt_block(block128: int, round_keys: List[int]) -> int:
    """Encrypt a 128-bit block using SEED algorithm."""
    # Split into two 64-bit halves
    left = (block128 >> 64) & 0xFFFFFFFF_FFFFFFFF
    right = block128 & 0xFFFFFFFF_FFFFFFFF
    
    # Split each half into two 32-bit words
    L0 = (left >> 32) & 0xFFFFFFFF
    L1 = left & 0xFFFFFFFF
    R0 = (right >> 32) & 0xFFFFFFFF
    R1 = right & 0xFFFFFFFF
    
    # Apply 16 rounds
    for i in range(16):
        # Apply F function and key mixing
        T0 = R0 ^ R1
        T1 = T0 ^ round_keys[i]
        T1 = seed_f(T1, round_keys[i])
        
        # Update state
        T0 = L0 ^ T1
        L0 = R0
        R0 = L1 ^ T1
        L1 = R1
        R1 = T0
    
    # Final swap and recombine
    return ((R0 << 96) | (R1 << 64) | (L0 << 32) | L1)

# Add this function to implement RC5 encryption
def rc5_encrypt_block(block64: int, S: List[int], w: int = 32, r: int = 12) -> int:
    """
    RC5 block encryption.
    
    Args:
        block64: 64-bit block to encrypt
        S: Round key array
        w: Word size in bits (default 32)
        r: Number of rounds (default 12)
    
    Returns:
        Encrypted 64-bit block
    """
    # Split block into two w-bit words
    A, B = split_into_two_w(block64, w)
    mask = (1 << w) - 1
    
    # Initial key mixing
    A = (A + S[0]) & mask
    B = (B + S[1]) & mask
    
    # r rounds of encryption
    for i in range(1, r+1):
        A = (rotate_left((A ^ B), B & (w-1), w) + S[2*i]) & mask
        B = (rotate_left((B ^ A), A & (w-1), w) + S[2*i+1]) & mask
    
    # Combine the two words back into a block
    return combine_two_w(A, B, w)

# Add RC6 encryption function
def rc6_encrypt_block(block128: int, S: List[int], w: int = 32, r: int = 20) -> int:
    """
    RC6 block encryption.
    
    Args:
        block128: 128-bit block to encrypt
        S: Round key array
        w: Word size in bits (default 32)
        r: Number of rounds (default 20)
    
    Returns:
        Encrypted 128-bit block
    """
    # Split block into four w-bit words
    A, B, C, D = split_into_four_w(block128, w)
    mask = (1 << w) - 1
    
    # Initial key addition
    B = (B + S[0]) & mask
    D = (D + S[1]) & mask
    
    # r rounds
    for i in range(1, r+1):
        t = rotate_left((B * (2*B + 1)) & mask, 5, w)
        u = rotate_left((D * (2*D + 1)) & mask, 5, w)
        A = (rotate_left(A ^ t, u & (w-1), w) + S[2*i]) & mask
        C = (rotate_left(C ^ u, t & (w-1), w) + S[2*i+1]) & mask
        A, B, C, D = B, C, D, A  # Rotate variables
    
    # Final key addition
    A = (A + S[2*r+2]) & mask
    C = (C + S[2*r+3]) & mask
    
    # Combine the four words back into a block
    return combine_four_w(A, B, C, D, w)

# ========================
# Decryption Blocks
# ========================

def aes_decrypt_block(cipher_bytes: bytes, master_key: bytes) -> bytes:
    round_keys = aes_key_expansion(master_key)
    state = list(cipher_bytes)
    state = add_round_key(state, round_keys[-1])
    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)
    for rk in reversed(round_keys[1:-1]):
        state = add_round_key(state, rk)
        state = inv_mix_columns(state)
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)
    state = add_round_key(state, round_keys[0])
    return bytes(state)

def des_decrypt_block(block64: int, master_key64: int) -> int:
    subkeys = des_key_schedule(master_key64)
    state = initial_permutation(block64)
    state = feistel_cipher(state, subkeys[::-1], des_f, half_size=32)
    return final_permutation(state)

def triple_des_decrypt_block(block64: int, keys: Tuple[int,int,int]) -> int:
    k1,k2,k3 = keys
    p1 = des_decrypt_block(block64, k3)
    p2 = des_encrypt_block(p1,     k2)
    return des_decrypt_block(p2,    k1)

def blowfish_decrypt_block(block64: int, key_bytes: bytes, p_array: List[int], s_boxes: List[List[int]]) -> int:
    # Split the 64-bit block into two 32-bit halves
    left = (block64 >> 32) & 0xFFFFFFFF
    right = block64 & 0xFFFFFFFF
    
    # Apply the operations in reverse order
    left ^= p_array[17]
    right ^= p_array[16]
    
    # The 16 rounds in reverse
    for i in range(15, -1, -1):
        left, right = right, left
        right ^= blowfish_f(left, s_boxes)
        left ^= p_array[i]
    
    # Combine the halves back into a 64-bit block
    return (left << 32) | right

def twofish_decrypt_block(block128: int, master_key: bytes) -> int:
    subkeys, q_boxes, mds = twofish_key_schedule(master_key, num_rounds=16)
    state = block128 ^ (subkeys[-2] << 64 | subkeys[-1])
    for rk in reversed(subkeys[2:18]):
        Fout = twofish_f(state & ((1 << 64) - 1), q_boxes, mds, rk)
        state = ((state >> 64) ^ Fout) | ((state & ((1 << 64) - 1)) << 64)
    return state ^ (subkeys[0] << 64 | subkeys[1])

def serpent_decrypt_block(cipher128: int, round_keys: List[int]) -> int:
    # Reverse final key XOR
    state = cipher128 ^ round_keys[-1]
    # Build inverse linear permutation
    inv_p = [0] * len(serpent_linear_p)
    for i, pos in enumerate(serpent_linear_p):
        inv_p[pos-1] = i+1
    # Build inverse S-box layers
    inv_sboxes = []
    for sbox in serpent_sboxes:
        inv = [0]*16
        for i,v in enumerate(sbox): inv[v] = i
        inv_sboxes.append(inv)
    # Run rounds in reverse
    for rk, inv_sbox in zip(reversed(round_keys[:-1]), reversed(inv_sboxes)):
        # inverse linear transform
        state = apply_permutation(state, inv_p, 128)
        # inverse S-box layer nibblewise
        state = apply_sbox_layer(state, [inv_sbox], block_size=128, nibble_size=4)
        # XOR round key
        state ^= rk
    return state

def camellia_decrypt_block(block128: int, round_keys: List[int], fl_keys: List[int]) -> int:
    # Undo post-whitening
    state = block128 ^ round_keys[-1]
    # 18 rounds in reverse
    for i in range(18, 0, -1):
        if i in (6, 12):
            # FL/FL^-1 is involutive
            state = camellia_fl(state, fl_keys[i//6 - 1])
        # Feistel step reverse is same as forward with same subkey order
        state = feistel_rounds_step(state, round_keys[i], camellia_f)
    # Undo pre-whitening
    return state ^ round_keys[0]

def cast128_decrypt_block(block64: int, round_keys: Any) -> int:
    return feistel_cipher(block64, list(round_keys)[::-1], cast128_f, half_size=32)

def idea_decrypt_block(block64: int, round_keys: List[int]) -> int:
    """Decrypt a 64-bit block using IDEA algorithm."""
    # For IDEA decryption, we need to:
    # 1. Create inverted round keys (reverse order and invert operations)
    # 2. Apply the same algorithm as encryption with these inverted keys
    
    # Create decryption subkeys by inverting encryption subkeys
    decrypt_keys = []
    
    # Output transformation keys (first 4 for decryption)
    # Multiplicative inverse mod 2^16+1
    decrypt_keys.append(modular_multiplicative_inverse(round_keys[-4], 0x10001))  # k49 = 1/k46
    # Additive inverse mod 2^16
    decrypt_keys.append((0x10000 - round_keys[-3]) & 0xFFFF)   # k50 = -k47
    decrypt_keys.append((0x10000 - round_keys[-2]) & 0xFFFF)   # k51 = -k48
    decrypt_keys.append(modular_multiplicative_inverse(round_keys[-1], 0x10001))  # k52 = 1/k49
    
    # Round keys for decryption (reversing and inverting)
    for i in range(7, -1, -1):  # for rounds 7 down to 0
        idx = i * 6  # Start of round keys
        
        # For each group of 6 round keys:
        decrypt_keys.append(modular_multiplicative_inverse(round_keys[idx], 0x10001))  # 1/k_i1
        decrypt_keys.append((0x10000 - round_keys[idx+2]) & 0xFFFF)  # -k_i3
        decrypt_keys.append((0x10000 - round_keys[idx+1]) & 0xFFFF)  # -k_i2
        decrypt_keys.append(modular_multiplicative_inverse(round_keys[idx+3], 0x10001))  # 1/k_i4
        decrypt_keys.append(round_keys[idx+4])  # k_i5 (remains the same)
        decrypt_keys.append(round_keys[idx+5])  # k_i6 (remains the same)
    
    # Split into four 16-bit blocks
    x1, x2, x3, x4 = split_into_four_16bit(block64)
    
    # Apply 8 rounds of decryption with inverted keys
    for round in range(8):
        # Extract 6 subkeys for this round
        k = decrypt_keys[round*6+4:(round+1)*6+4]
        # Apply round function
        x1, x2, x3, x4 = idea_round(x1, x2, x3, x4, k)
    
    # Final output transformation with first 4 inverted keys
    x1, x2, x3, x4 = idea_output_transform(x1, x2, x3, x4, decrypt_keys[:4])
    
    # Combine into 64-bit result
    return (x1 << 48) | (x2 << 32) | (x3 << 16) | x4

# Helper function for IDEA decryption
def modular_multiplicative_inverse(x: int, modulus: int) -> int:
    """Calculate multiplicative inverse of x modulo modulus."""
    if x == 0:
        return 0
    
    # Extended Euclidean Algorithm
    a, b = x, modulus
    y1, y2 = 1, 0
    
    while b > 0:
        q, r = divmod(a, b)
        y1, y2 = y2, y1 - q * y2
        a, b = b, r
    
    # Make sure we return a positive value
    return y1 % modulus

def rc5_decrypt_block(block64: int, S: List[int], w: int = 32, r: int = 12) -> int:
    A, B = split_into_two_w(block64, w)
    for i in range(r, 0, -1):
        B = rotate_left(B - S[2*i+1] & ((1<<w)-1), A & (w-1), w) ^ A
        A = rotate_left(A - S[2*i] & ((1<<w)-1), B & (w-1), w) ^ B
    A = (A - S[0]) & ((1<<w)-1)
    B = (B - S[1]) & ((1<<w)-1)
    return combine_two_w(A, B, w)

def rc6_decrypt_block(block128: int, S: List[int], w: int = 32, r: int = 20) -> int:
    A, B, C, D = split_into_four_w(block128, w)
    C = (C - S[2*r+3]) & ((1<<w)-1)
    A = (A - S[2*r+2]) & ((1<<w)-1)
    for i in range(r, 0, -1):
        A, B, C, D = D, A, B, C
        u = rotate_left(D*(2*D+1) & ((1<<w)-1), 5, w)
        t = rotate_left(B*(2*B+1) & ((1<<w)-1), 5, w)
        C = rotate_left(C - S[2*i+1] & ((1<<w)-1), t & (w-1), w) ^ u
        A = rotate_left(A - S[2*i] & ((1<<w)-1), u & (w-1), w) ^ t
    D = (D - S[1]) & ((1<<w)-1)
    B = (B - S[0]) & ((1<<w)-1)
    return combine_four_w(A, B, C, D, w)

def seed_decrypt_block(block64: int, round_keys: List[int]) -> int:
    return feistel_cipher(block64, list(round_keys)[::-1], seed_f, half_size=32)

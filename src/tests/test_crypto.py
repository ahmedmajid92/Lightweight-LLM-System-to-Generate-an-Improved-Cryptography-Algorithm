from Components import blowfish_key_schedule, twofish_key_schedule, serpent_key_schedule
from AlgorithmsBlock import (
    bytes_to_int, int_to_bytes,
    aes_encrypt_block, aes_decrypt_block,
    des_encrypt_block, des_decrypt_block,
    triple_des_encrypt_block, triple_des_decrypt_block,
    blowfish_encrypt_block, blowfish_decrypt_block,
    twofish_encrypt_block, twofish_decrypt_block,
    serpent_encrypt_block, serpent_decrypt_block,
    camellia_encrypt_block, camellia_decrypt_block,
    cast128_encrypt_block, cast128_decrypt_block,
    idea_encrypt_block, idea_decrypt_block,
    rc5_encrypt_block, rc5_decrypt_block,
    rc6_encrypt_block, rc6_decrypt_block,
    seed_encrypt_block, seed_decrypt_block,
)

# Sample data and keys
AES_KEY = b"0123456789ABCDEF"           # 16 bytes
AES_PLAIN = b"Hello, AES!!!!!!!!!"      # 16 bytes plaintext

DES_KEY = 0x0123456789ABCDEF             # 64-bit key
DES_PLAIN = b"ABCDEFGH"                # 8-byte block

TDEA_KEYS = (0x0123456789ABCDEF, 0xFEDCBA9876543210, 0x0011223344556677)

BLOWFISH_KEY = b"MyBlowfishKey!"
BLOWFISH_PLAIN = b"abcdefgh"

TWOFISH_KEY = b"TwofishSecretKey123"      # example length
TWOFISH_PLAIN = b"TwofishBlock128!!"      # 16 bytes

SERPENT_KEYS = [i for i in range(33)]     # placeholder round keys
SERPENT_PLAIN = bytes(range(16))          # 128-bit block

CAMELLIA_KEYS = [i for i in range(20)]    # placeholder round keys
CAMELLIA_FL_KEYS = [i for i in range(2)]  # placeholder FL-keys
CAMELLIA_PLAIN = bytes(range(16))

CAST128_KEYS = [i for i in range(17)]      # placeholder round keys
CAST128_PLAIN = b"Cast128!"

IDEA_KEYS = [i for i in range(52)]        # 52 subkeys (6*8+4)
IDEA_PLAIN = b"IDEA1234"

RC5_S = [i for i in range(34)]            # S-array, w=32, r=12 -> 2*(r+1)=26; extra for safety
RC5_PLAIN = b"RC5TEST!"

RC6_S = [i for i in range(44)]            # S-array, w=32, r=20 -> 2*(r+2)=44
RC6_PLAIN = b"RC6TEST!"

SEED_KEYS = [i for i in range(17)]        # 16 round keys + placeholder
SEED_PLAIN = b"Seed1234"


def test_aes():
    print("AES Test:")
    ct = aes_encrypt_block(AES_PLAIN, AES_KEY)
    pt = aes_decrypt_block(ct, AES_KEY)
    print(" Plain:", AES_PLAIN)
    print("Cipher:", ct)
    print("Recovered:", pt)
    assert pt == AES_PLAIN


def test_des():
    print("DES Test:")
    blk = bytes_to_int(DES_PLAIN)
    ct = des_encrypt_block(blk, DES_KEY)
    pt = des_decrypt_block(ct, DES_KEY)
    recovered = int_to_bytes(pt, 8)
    print(" Plain:", DES_PLAIN)
    print("Cipher:", hex(ct))
    print("Recovered:", recovered)
    assert recovered == DES_PLAIN


def test_3des():
    print("3DES Test:")
    blk = bytes_to_int(DES_PLAIN)
    ct = triple_des_encrypt_block(blk, TDEA_KEYS)
    pt = triple_des_decrypt_block(ct, TDEA_KEYS)
    recovered = int_to_bytes(pt, 8)
    print("Recovered:", recovered)
    assert recovered == DES_PLAIN


def test_blowfish():
    print("Blowfish Test:")
    blk = bytes_to_int(BLOWFISH_PLAIN)
    p_array, s_boxes = blowfish_key_schedule(BLOWFISH_KEY)
    ct = blowfish_encrypt_block(blk, BLOWFISH_KEY, p_array, s_boxes)
    pt = blowfish_decrypt_block(ct, BLOWFISH_KEY, p_array, s_boxes)
    recovered = int_to_bytes(pt, 8)
    print("Recovered:", recovered)
    assert recovered == BLOWFISH_PLAIN

def test_twofish():
    ks = twofish_key_schedule(TWOFISH_KEY)
    blk = bytes_to_int(TWOFISH_PLAIN)
    ct = twofish_encrypt_block(blk, ks)
    pt = twofish_decrypt_block(ct, TWOFISH_KEY)
    recovered = int_to_bytes(pt, 16)
    assert recovered == TWOFISH_PLAIN


def test_serpent():
    blk = bytes_to_int(SERPENT_PLAIN)
    ct = serpent_encrypt_block(blk, SERPENT_KEYS)
    pt = serpent_decrypt_block(ct, SERPENT_KEYS)
    recovered = int_to_bytes(pt, 16)
    assert recovered == SERPENT_PLAIN


def test_camellia():
    blk = bytes_to_int(CAMELLIA_PLAIN)
    ct = camellia_encrypt_block(blk, CAMELLIA_KEYS, CAMELLIA_FL_KEYS)
    pt = camellia_decrypt_block(ct, CAMELLIA_KEYS, CAMELLIA_FL_KEYS)
    recovered = int_to_bytes(pt, 16)
    assert recovered == CAMELLIA_PLAIN


def test_cast128():
    blk = bytes_to_int(CAST128_PLAIN)
    ct = cast128_encrypt_block(blk, CAST128_KEYS)
    pt = cast128_decrypt_block(ct, CAST128_KEYS)
    recovered = int_to_bytes(pt, 8)
    assert recovered == CAST128_PLAIN


def test_idea():
    blk = bytes_to_int(IDEA_PLAIN)
    ct = idea_encrypt_block(blk, IDEA_KEYS)
    pt = idea_decrypt_block(ct, IDEA_KEYS)
    recovered = int_to_bytes(pt, 8)
    assert recovered == IDEA_PLAIN


def test_rc5():
    blk = bytes_to_int(RC5_PLAIN)
    ct = rc5_encrypt_block(blk, RC5_S)
    pt = rc5_decrypt_block(ct, RC5_S)
    recovered = int_to_bytes(pt, 8)
    assert recovered == RC5_PLAIN


def test_rc6():
    blk = bytes_to_int(RC6_PLAIN)
    ct = rc6_encrypt_block(blk, RC6_S)
    pt = rc6_decrypt_block(ct, RC6_S)
    recovered = int_to_bytes(pt, 16)
    assert recovered == RC6_PLAIN


def test_seed():
    blk = bytes_to_int(SEED_PLAIN)
    ct = seed_encrypt_block(blk, SEED_KEYS)
    pt = seed_decrypt_block(ct, SEED_KEYS)
    recovered = int_to_bytes(pt, 8)
    assert recovered == SEED_PLAIN


def safe_test(name: str, fn: callable, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        print(f"{name}: ✔️")
    except NotImplementedError:
        print(f"{name}: ✖️ not implemented yet")
    except AssertionError:
        print(f"{name}: ❌ wrong result")
    except Exception as e:
        print(f"{name}: ⚠️ error {e!r}")
        
        
if __name__ == "__main__":
    safe_test("AES", aes_encrypt_block, AES_PLAIN, AES_KEY)
    safe_test("AES-DEC", aes_decrypt_block, AES_PLAIN, AES_KEY)
    safe_test("DES", des_encrypt_block, bytes_to_int(DES_PLAIN), DES_KEY)
    safe_test("DES-DEC", des_decrypt_block, bytes_to_int(DES_PLAIN), DES_KEY)
    safe_test("3DES", triple_des_encrypt_block, bytes_to_int(DES_PLAIN), TDEA_KEYS)
    safe_test("3DES-DEC", triple_des_decrypt_block, bytes_to_int(DES_PLAIN), TDEA_KEYS)

    # Additional algorithm tests
    safe_test("Blowfish",     blowfish_encrypt_block, bytes_to_int(BLOWFISH_PLAIN), BLOWFISH_KEY, *blowfish_key_schedule(BLOWFISH_KEY))
    safe_test("Blowfish-DEC", blowfish_decrypt_block, bytes_to_int(BLOWFISH_PLAIN), BLOWFISH_KEY, *blowfish_key_schedule(BLOWFISH_KEY))

    safe_test("Twofish",     twofish_encrypt_block, bytes_to_int(TWOFISH_PLAIN), TWOFISH_KEY)
    safe_test("Twofish-DEC", twofish_decrypt_block, bytes_to_int(TWOFISH_PLAIN), TWOFISH_KEY)

    safe_test("Serpent",     serpent_encrypt_block, bytes_to_int(SERPENT_PLAIN), SERPENT_KEYS)
    safe_test("Serpent-DEC", serpent_decrypt_block, bytes_to_int(SERPENT_PLAIN), SERPENT_KEYS)

    safe_test("Camellia",     camellia_encrypt_block, bytes_to_int(CAMELLIA_PLAIN), CAMELLIA_KEYS, CAMELLIA_FL_KEYS)
    safe_test("Camellia-DEC", camellia_decrypt_block, bytes_to_int(CAMELLIA_PLAIN), CAMELLIA_KEYS, CAMELLIA_FL_KEYS)

    safe_test("CAST128",    cast128_encrypt_block, bytes_to_int(CAST128_PLAIN), CAST128_KEYS)
    safe_test("CAST128-DEC", cast128_decrypt_block, bytes_to_int(CAST128_PLAIN), CAST128_KEYS)

    safe_test("IDEA",    idea_encrypt_block, bytes_to_int(IDEA_PLAIN), IDEA_KEYS)
    safe_test("IDEA-DEC", idea_decrypt_block, bytes_to_int(IDEA_PLAIN), IDEA_KEYS)

    safe_test("RC5",    rc5_encrypt_block, bytes_to_int(RC5_PLAIN), RC5_S)
    safe_test("RC5-DEC", rc5_decrypt_block, bytes_to_int(RC5_PLAIN), RC5_S)

    safe_test("RC6",    rc6_encrypt_block, bytes_to_int(RC6_PLAIN), RC6_S)
    safe_test("RC6-DEC", rc6_decrypt_block, bytes_to_int(RC6_PLAIN), RC6_S)

    safe_test("SEED",    seed_encrypt_block, bytes_to_int(SEED_PLAIN), SEED_KEYS)
    safe_test("SEED-DEC", seed_decrypt_block, bytes_to_int(SEED_PLAIN), SEED_KEYS)

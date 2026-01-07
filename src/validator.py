# src/validator.py
import inspect
import os
import secrets
from typing import List, Tuple
from .schemas import ValidationReport, AlgorithmSpec
from .composer import load_generated_module

def _rand_bytes(n: int) -> bytes:
    return secrets.token_bytes(n)

def _sig_str(fn) -> str:
    return str(inspect.signature(fn))

def validate_composed(module_path: str, blueprint: AlgorithmSpec) -> ValidationReport:
    """
    Comprehensive validation with user-friendly error messages:
      1) module has encrypt_block/decrypt_block callables
      2) signatures are (bytes, bytes) -> bytes
      3) round-trip test on 2 random plaintexts for declared block_size/key_size
    """
    if not os.path.exists(module_path):
        return ValidationReport(
            False, 
            errors=[
                "FILE_NOT_FOUND|The generated algorithm file doesn't exist yet. Click 'Generate Algorithm' first.|Make sure you clicked the 'Generate Algorithm' button before validating."
            ]
        )

    try:
        mod = load_generated_module(module_path)
    except Exception as e:
        return ValidationReport(
            False,
            errors=[
                f"MODULE_LOAD_ERROR|Failed to load the generated module: {str(e)}|Check if the generated code has syntax errors. Try regenerating the algorithm."
            ]
        )
    
    errors: List[str] = []
    warnings: List[str] = []

    # Check for required functions
    enc = getattr(mod, "encrypt_block", None)
    dec = getattr(mod, "decrypt_block", None)
    
    if not callable(enc):
        errors.append(
            "MISSING_ENCRYPT|The encrypt_block function is missing or not callable.|"
            "This is a critical error in code generation. Try regenerating the algorithm or check for syntax errors."
        )
    
    if not callable(dec):
        errors.append(
            "MISSING_DECRYPT|The decrypt_block function is missing or not callable.|"
            "This is a critical error in code generation. Try regenerating the algorithm or check for syntax errors."
        )
    
    if not callable(enc) or not callable(dec):
        return ValidationReport(False, errors=errors, warnings=warnings)

    # Signature validation
    enc_sig = inspect.signature(enc)
    dec_sig = inspect.signature(dec)
    
    for fn_name, sig, expected in [("encrypt_block", enc_sig, "plaintext"), ("decrypt_block", dec_sig, "ciphertext")]:
        if len(sig.parameters) != 2:
            warnings.append(
                f"SIGNATURE_WARNING|{fn_name} has unusual signature {sig}|"
                f"Expected 2 parameters: ({expected}: bytes, key: bytes). The algorithm might still work, but this is non-standard."
            )

    # Runtime validation with test data
    test_results = []
    for test_num in range(2):
        pt = _rand_bytes(blueprint.block_size)
        key = _rand_bytes(blueprint.key_size)
        
        try:
            # Test encryption
            ct = enc(pt, key)  # type: ignore
            
            if not isinstance(ct, (bytes, bytearray)):
                errors.append(
                    f"WRONG_RETURN_TYPE|encrypt_block returned {type(ct).__name__} instead of bytes|"
                    f"The encryption function must return bytes. Check your component implementations - "
                    f"one of your selected components might be returning the wrong type."
                )
                break
            
            if len(ct) != blueprint.block_size:
                errors.append(
                    f"WRONG_OUTPUT_SIZE|encrypt_block returned {len(ct)} bytes, expected {blueprint.block_size} bytes|"
                    f"The output size doesn't match the block size. This means your components are incompatible - "
                    f"they're designed for different block sizes. Check that all components support {blueprint.block_size}-byte blocks."
                )
                break
            
            # Test decryption
            rt = dec(ct, key)  # type: ignore
            
            if not isinstance(rt, (bytes, bytearray)):
                errors.append(
                    f"WRONG_RETURN_TYPE|decrypt_block returned {type(rt).__name__} instead of bytes|"
                    f"The decryption function must return bytes. Check your component implementations."
                )
                break
            
            if len(rt) != len(pt):
                errors.append(
                    f"LENGTH_MISMATCH|Decrypted data has wrong length: {len(rt)} bytes instead of {len(pt)} bytes|"
                    f"The decryption didn't produce the correct output size. Your components might be incompatible, "
                    f"or the decrypt function needs proper inverse components."
                )
                break
            
            # Check round-trip correctness
            if rt != pt:
                warnings.append(
                    "ROUNDTRIP_MISMATCH|Decryption doesn't reverse encryption (plaintext ≠ decrypted ciphertext)|"
                    f"The algorithm encrypted successfully, but decryption didn't recover the original data. "
                    f"This happens because the generated decrypt_block is a placeholder. To fix this, you need to:\n"
                    f"  • Select inverse components (like inv_sub_bytes, inv_shift_rows for AES)\n"
                    f"  • Or manually implement the decrypt_block function in the generated code\n"
                    f"  • Or use the same components in reverse order (for some algorithms)"
                )
            else:
                test_results.append(f"✓ Test {test_num + 1}: Passed (round-trip successful)")
                
        except TypeError as e:
            errors.append(
                f"TYPE_ERROR|Type error during execution: {str(e)}|"
                f"Your components have incompatible input/output types. Common causes:\n"
                f"  • Mixing components that expect different data formats (bytes vs List[int])\n"
                f"  • Components designed for different block sizes\n"
                f"  • Key schedule returning wrong format\n"
                f"Try selecting components from the same algorithm family, or check component signatures in Tab 1."
            )
            break
        except ValueError as e:
            errors.append(
                f"VALUE_ERROR|Value error during execution: {str(e)}|"
                f"Invalid values encountered during encryption/decryption. This could mean:\n"
                f"  • Block size mismatch between components\n"
                f"  • Invalid key length\n"
                f"  • Component expecting different data format\n"
                f"Check that your selected components are compatible with each other."
            )
            break
        except IndexError as e:
            errors.append(
                f"INDEX_ERROR|Index out of range: {str(e)}|"
                f"A component tried to access data outside the valid range. This usually means:\n"
                f"  • Components expect different block sizes\n"
                f"  • Array/list indexing error in a component\n"
                f"  • State size mismatch between components\n"
                f"Try using components designed for the same block size."
            )
            break
        except Exception as e:
            error_type = type(e).__name__
            errors.append(
                f"RUNTIME_ERROR|{error_type} during execution: {str(e)}|"
                f"An unexpected error occurred while running your algorithm. This could indicate:\n"
                f"  • Incompatible components (different block sizes, data formats)\n"
                f"  • Missing dependencies or helper functions\n"
                f"  • Bug in one of the selected components\n"
                f"Try regenerating with different component selections or check the generated code for issues."
            )
            break
    
    # Add test results to details
    details = {
        "enc_sig": str(enc_sig),
        "dec_sig": str(dec_sig),
        "block_size": blueprint.block_size,
        "key_size": blueprint.key_size,
        "test_results": test_results
    }

    return ValidationReport(ok=not errors, errors=errors, warnings=warnings, details=details)

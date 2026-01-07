# src/component_registry.py
import inspect
import importlib
import re
from typing import Dict, List, Optional
from .schemas import ComponentSpec

# Loads Components.py once and introspects functions.
_COMPONENTS_MOD = None

def _load_components_module():
    global _COMPONENTS_MOD
    if _COMPONENTS_MOD is None:
        _COMPONENTS_MOD = importlib.import_module("src.Components")
    return _COMPONENTS_MOD

def _guess_algorithm_and_stage(fn_name: str, doc: str) -> (Optional[str], Optional[str]):
    """
    Heuristics:
      - prefix-based: aes_key_expansion -> algorithm=aes, stage=key_expansion
      - known functions: sub_bytes -> algorithm=AES, stage=sub_bytes
      - docstring fallback: "Algorithm: AES", "Stage: SubBytes"
    """
    # Map of known algorithm-specific functions without prefixes
    known_functions = {
        # AES functions (in Components.py without aes_ prefix)
        "sub_bytes": ("AES", "sub_bytes"),
        "inv_sub_bytes": ("AES", "sub_bytes"),
        "shift_rows": ("AES", "shift_rows"),
        "inv_shift_rows": ("AES", "shift_rows"),
        "mix_columns": ("AES", "mix_columns"),
        "inv_mix_columns": ("AES", "mix_columns"),
        "add_round_key": ("AES", "add_round_key"),
        "aes_key_expansion": ("AES", "key_schedule"),
        # DES functions
        "des_key_schedule": ("DES", "key_schedule"),
        "des_f": ("DES", "round_function"),
        "initial_permutation": ("DES", "permute"),
        "final_permutation": ("DES", "permute"),
        "triple_des_encrypt": ("3DES", "encrypt"),
        # Blowfish functions
        "blowfish_f": ("BLOWFISH", "round_function"),
        "blowfish_key_schedule": ("BLOWFISH", "key_schedule"),
        # Twofish functions
        "twofish_key_schedule": ("TWOFISH", "key_schedule"),
        "twofish_f": ("TWOFISH", "round_function"),
        # Serpent functions
        "serpent_key_schedule": ("SERPENT", "key_schedule"),
        # Camellia functions
        "camellia_f": ("CAMELLIA", "round_function"),
        "camellia_fl": ("CAMELLIA", "fl_function"),
        # CAST-128 functions
        "cast128_f": ("CAST-128", "round_function"),
        # IDEA functions
        "idea_mul": ("IDEA", "multiply"),
        "idea_add": ("IDEA", "add"),
        "idea_round": ("IDEA", "round_function"),
        "idea_output_transform": ("IDEA", "output_transform"),
        # SEED functions (components only, not main encrypt/decrypt)
        "g_function": ("SEED", "g_function"),
        "seed_f": ("SEED", "round_function"),
        # RC5 functions
        "rc5_key_schedule": ("RC5", "key_schedule"),
        # RC6 functions
        "rc6_key_schedule": ("RC6", "key_schedule"),
    }
    
    # Check known functions first
    if fn_name in known_functions:
        return known_functions[fn_name]
    
    # Try prefix-based matching
    m = re.match(r"([a-z0-9]+)_(.+)", fn_name)
    alg = stage = None
    if m:
        alg, stage = m.group(1), m.group(2)
    
    # normalize common names
    norm = {
        "subbytes": "sub_bytes",
        "mixcolumns": "mix_columns",
        "shiftrows": "shift_rows",
        "addroundkey": "add_round_key",
        "keyschedule": "key_schedule",
        "keyexpansion": "key_schedule",
        "roundfunction": "round_function",
    }
    if stage:
        s = stage.lower().replace("-", "_")
        stage = norm.get(s, s)

    # docstring hints
    if doc:
        alg_m = re.search(r"(?i)algorithm\s*:\s*([A-Za-z0-9\-_/ ]+)", doc)
        stage_m = re.search(r"(?i)stage\s*:\s*([A-Za-z0-9\-_/ ]+)", doc)
        if alg_m:
            alg = alg_m.group(1).strip().replace(" ", "")
        if stage_m:
            stage = stage_m.group(1).strip().lower().replace(" ", "_")
    if alg:
        alg = alg.upper()
    return alg, stage

def _io_hints_from_signature(sig: inspect.Signature):
    """Turn parameters/return into soft IO tokens for validator & UI."""
    inputs = [p.name for p in sig.parameters.values()]
    # return hints are not visible; keep symbolic list empty, validator will probe at runtime
    return inputs, []

def scan_components() -> Dict[str, List[ComponentSpec]]:
    """
    Returns: dict[algorithm_name] -> list[ComponentSpec]
    Includes ALL functions from Components.py, even utilities and helpers
    """
    mod = _load_components_module()
    alg_map: Dict[str, List[ComponentSpec]] = {}
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if obj.__module__ != mod.__name__:
            continue
        doc = inspect.getdoc(obj) or ""
        alg, stage = _guess_algorithm_and_stage(name, doc)
        
        # If can't classify, put in UTILITY category so it's still available
        if not alg:
            alg = "UTILITY"
            stage = "helper"
        
        sig = inspect.signature(obj)
        inputs, outputs = _io_hints_from_signature(sig)
        comp = ComponentSpec(
            name=name,
            algorithm=alg,
            stage=stage,
            fn_ref=obj,
            doc=doc,
            inputs=inputs,
            outputs=outputs,
            signature_str=str(sig),
        )
        alg_map.setdefault(alg, []).append(comp)
    # sort for stable UI
    for k in alg_map:
        alg_map[k].sort(key=lambda c: (c.stage or "", c.name))
    return alg_map

def list_algorithms() -> List[str]:
    return sorted(scan_components().keys())

def list_components_for_algorithm(algorithm: str) -> List[ComponentSpec]:
    return scan_components().get(algorithm.upper(), [])

def find_component_by_name(name: str) -> Optional[ComponentSpec]:
    for comps in scan_components().values():
        for c in comps:
            if c.name == name:
                return c
    return None

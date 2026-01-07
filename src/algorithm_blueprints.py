# src/algorithm_blueprints.py
from typing import Dict, List
from .schemas import AlgorithmSpec, StageSpec

# Canonical stage sets; keep this small & generic so it works with your repo immediately.
# If an algorithm in your repo doesn't use a stage, the composer will skip missing ones.
_COMMON_SPN_STAGES = [
    StageSpec("key_schedule", description="Generate round keys", expected_input_tokens=["key"], expected_output_tokens=["round_keys"]),
    StageSpec("sub_bytes",    description="Non-linear substitution layer", expected_input_tokens=["state"], expected_output_tokens=["state"]),
    StageSpec("shift_rows",   description="Row permutation", expected_input_tokens=["state"], expected_output_tokens=["state"]),
    StageSpec("mix_columns",  description="Column mixing (linear layer)", expected_input_tokens=["state"], expected_output_tokens=["state"]),
    StageSpec("add_round_key",description="XOR state with round key", expected_input_tokens=["state","round_key"], expected_output_tokens=["state"]),
]

_COMMON_FEISTEL_STAGES = [
    StageSpec("key_schedule",    description="Generate round subkeys", expected_input_tokens=["key"], expected_output_tokens=["subkeys"]),
    StageSpec("round_function",  description="Feistel F function", expected_input_tokens=["right","subkey"], expected_output_tokens=["fout"]),
    StageSpec("permute", required=False, description="Optional initial/final permutation", expected_input_tokens=["block"], expected_output_tokens=["block"]),
]

# Complete set of algorithms from algorithms.json
_ALGOS: Dict[str, AlgorithmSpec] = {
    "AES": AlgorithmSpec(name="AES", block_size=16, key_size=16, structure="SPN", rounds=10, stages=_COMMON_SPN_STAGES),
    "DES": AlgorithmSpec(name="DES", block_size=8, key_size=8, structure="Feistel", rounds=16, stages=_COMMON_FEISTEL_STAGES),
    "3DES": AlgorithmSpec(name="3DES", block_size=8, key_size=24, structure="Feistel", rounds=48, stages=_COMMON_FEISTEL_STAGES),
    "BLOWFISH": AlgorithmSpec(name="BLOWFISH", block_size=8, key_size=16, structure="Feistel", rounds=16, stages=_COMMON_FEISTEL_STAGES),
    "TWOFISH": AlgorithmSpec(name="TWOFISH", block_size=16, key_size=16, structure="Feistel-like", rounds=16, stages=_COMMON_FEISTEL_STAGES),
    "SERPENT": AlgorithmSpec(name="SERPENT", block_size=16, key_size=16, structure="SPN", rounds=32, stages=_COMMON_SPN_STAGES),
    "CAMELLIA": AlgorithmSpec(name="CAMELLIA", block_size=16, key_size=16, structure="Feistel-like", rounds=18, stages=_COMMON_FEISTEL_STAGES),
    "CAST-128": AlgorithmSpec(name="CAST-128", block_size=8, key_size=16, structure="Feistel", rounds=16, stages=_COMMON_FEISTEL_STAGES),
    "IDEA": AlgorithmSpec(name="IDEA", block_size=8, key_size=16, structure="Lai-Massey", rounds=8, stages=_COMMON_FEISTEL_STAGES),
    "RC5": AlgorithmSpec(name="RC5", block_size=8, key_size=16, structure="Feistel-like", rounds=12, stages=_COMMON_FEISTEL_STAGES),
    "RC6": AlgorithmSpec(name="RC6", block_size=16, key_size=16, structure="Feistel-like", rounds=20, stages=_COMMON_SPN_STAGES),
    "SEED": AlgorithmSpec(name="SEED", block_size=16, key_size=16, structure="Feistel", rounds=16, stages=_COMMON_FEISTEL_STAGES),
}

def get_blueprint(algorithm: str) -> AlgorithmSpec:
    return _ALGOS.get(algorithm.upper(), AlgorithmSpec(
        name=algorithm.upper(), block_size=16, key_size=16, structure="SPN", rounds=10, stages=_COMMON_SPN_STAGES
    ))

def list_blueprints() -> List[str]:
    return sorted(_ALGOS.keys())

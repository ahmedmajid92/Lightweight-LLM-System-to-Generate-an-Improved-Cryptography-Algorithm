# src/schemas.py
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple

@dataclass
class ComponentSpec:
    name: str
    algorithm: str                 # e.g., "AES", "Serpent"
    stage: Optional[str] = None    # e.g., "sub_bytes", "mix_columns"
    fn_ref: Optional[Callable] = None
    doc: str = ""
    inputs: List[str] = field(default_factory=list)   # symbolic names (for UI & validation)
    outputs: List[str] = field(default_factory=list)
    signature_str: str = ""        # pretty-printed signature

@dataclass
class StageSpec:
    name: str
    required: bool = True
    description: str = ""
    # expected I/O "tokens" to help the validator (loose, but useful)
    expected_input_tokens: List[str] = field(default_factory=list)
    expected_output_tokens: List[str] = field(default_factory=list)

@dataclass
class AlgorithmSpec:
    name: str
    block_size: int                # in bytes
    key_size: int                  # in bytes (or max)
    structure: str                 # "SPN" or "Feistel" etc.
    rounds: int
    stages: List[StageSpec] = field(default_factory=list)

@dataclass
class CompositionRequest:
    base_algorithm: str
    # stage_name -> component name chosen
    selections: Dict[str, str]
    output_name: str               # name for the generated algorithm
    extra_rounds: Optional[int] = None

@dataclass
class CompositionResult:
    ok: bool
    module_code: str
    module_path: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

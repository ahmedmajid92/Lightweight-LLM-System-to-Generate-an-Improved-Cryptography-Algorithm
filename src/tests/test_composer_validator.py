# src/tests/test_composer_validator.py
import os
from src.algorithm_blueprints import get_blueprint
from src.schemas import CompositionRequest
from src.composer import compose
from src.validator import validate_composed

def test_compose_and_validate_smoke():
    bp = get_blueprint("AES")
    # minimal selection: only key_schedule + add_round_key to build a degenerate cipher
    selections = {
        "key_schedule": "aes_key_schedule",       # adapt to your functions
        "add_round_key": "aes_add_round_key",     # adapt to your functions
    }
    req = CompositionRequest(base_algorithm="AES", selections=selections, output_name="test_generated")
    result = compose(bp, req)
    assert result.ok, f"compose errors: {result.errors}"
    rep = validate_composed(result.module_path, bp)
    assert rep.ok or rep.warnings, "At least should run without hard errors"

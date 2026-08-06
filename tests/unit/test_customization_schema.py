import json
import pytest
from pathlib import Path

def test_customization_schema_structure():
    schema_path = Path("configs/customization_schema.json")
    assert schema_path.exists()
    
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "required_fields" in data
    assert "allowed_sizes" in data
    assert "gender" in data["required_fields"]
    assert "size" in data["required_fields"]
    assert "M" in data["allowed_sizes"]
    assert "XL" in data["allowed_sizes"]

# tests/test_utils/test_config.py
import os
import json
import pytest
from src.utils.config import load_config

def test_load_json_config(tmp_path):
    cfg = {'a': 1, 'b': 'text', 'nested': {'x': 10}}
    file = tmp_path / 'config.json'
    with open(file, 'w') as f:
        json.dump(cfg, f)

    loaded = load_config(str(file))
    assert isinstance(loaded, dict)
    assert loaded == cfg


def test_load_yaml_config(tmp_path):
    yaml_content = """
a: 1
b: text
nested:
  x: 10
"""
    file = tmp_path / 'config.yaml'
    with open(file, 'w') as f:
        f.write(yaml_content)

    loaded = load_config(str(file))
    assert isinstance(loaded, dict)
    assert loaded['a'] == 1
    assert loaded['b'] == 'text'
    assert 'nested' in loaded and loaded['nested']['x'] == 10


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / 'nope.json'))


def test_unsupported_extension(tmp_path):
    file = tmp_path / 'config.txt'
    with open(file, 'w') as f:
        f.write('a=1')
    with pytest.raises(ValueError):
        load_config(str(file))

@ pytest.mark.skipif(
    True,  # Simulate missing yaml library
    reason="Simulate no PyYAML"
)
def test_yaml_requires_pyyaml(tmp_path, monkeypatch):
    # Temporarily remove yaml
    import src.utils.config as cfgmod
    monkeypatch.setattr(cfgmod, 'yaml', None)
    file = tmp_path / 'config.yml'
    with open(file, 'w') as f:
        f.write('a: 1')
    with pytest.raises(ValueError):
        load_config(str(file))

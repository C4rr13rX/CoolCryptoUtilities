import json
from pathlib import Path

def test___init___foo_data():
    data_path = Path('runtime/c0d3r/foo_data.json')
    payload = json.loads(data_path.read_text())
    __import__('sat_core.sat.__init__')
    assert payload['foo'] == 'bar'

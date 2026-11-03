import os
import requests
import pytest

BASE = os.getenv('API_URL', 'http://localhost:8000')

def test_health():
    assert requests.get(BASE + '/health', timeout=10).json() == {'status': 'ok'}

def test_model_info():
    r = requests.get(BASE + '/model-info', timeout=10)
    assert r.status_code == 200
    assert set(r.json()['classes']) == {'yes', 'no', 'up', 'down'}

def test_accuracy_numeric():
    assert 0 <= requests.get(BASE + '/model-info', timeout=10).json()['accuracy'] <= 100

def test_model_size_positive():
    assert requests.get(BASE + '/model-info', timeout=10).json()['model_size_kb'] > 0

def test_invalid_extension():
    r = requests.post(BASE + '/predict', files={'file': ('x.txt', b'text')}, timeout=10)
    assert r.status_code == 400

def test_corrupt_wav():
    r = requests.post(BASE + '/predict', files={'file': ('x.wav', b'bad')}, timeout=10)
    assert r.status_code == 400

def test_missing_file():
    assert requests.post(BASE + '/predict', timeout=10).status_code == 422

def test_docs():
    assert requests.get(BASE + '/docs', timeout=10).status_code == 200

def test_openapi():
    assert '/predict' in requests.get(BASE + '/openapi.json', timeout=10).json()['paths']

def test_predict():
    from pathlib import Path
    sample = Path('samples/yes.wav')
    if not sample.exists():
        pytest.skip('samples generated during Docker training, not available on host')
    r = requests.post(BASE + '/predict', files={'file': ('yes.wav', sample.read_bytes())}, timeout=30)
    assert r.status_code == 200
    assert r.json()['label'] in {'yes', 'no', 'up', 'down'}

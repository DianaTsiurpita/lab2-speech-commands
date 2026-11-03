import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from .data import features
from .model import SpeechCNN

state = {}

@asynccontextmanager
async def lifespan(app):
    model_path = Path(os.getenv('MODEL_PATH', 'artifacts/model.pt'))
    labels = json.loads((model_path.parent / 'labels.json').read_text())
    metrics = json.loads((model_path.parent / 'metrics.json').read_text())
    torch.set_num_threads(2)
    model = SpeechCNN(len(labels))
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    state.update(model=model, labels=labels, metrics=metrics, model_path=model_path)
    yield
    state.clear()

app = FastAPI(title='Speech Commands API', version='1.0.0', lifespan=lifespan)

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/model-info')
def model_info():
    metrics = state['metrics']
    return {'architecture': metrics['architecture'], 'classes': state['labels'], 'accuracy': metrics['accuracy'], 'model_size_kb': round(state['model_path'].stat().st_size / 1024, 2), 'model_version': metrics['model_version']}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if not (file.filename or '').lower().endswith('.wav'):
        raise HTTPException(400, 'Only .wav files are supported')
    content = await file.read(5_000_001)
    if len(content) > 5_000_000:
        raise HTTPException(400, 'File exceeds 5 MB')
    try:
        audio, sr = sf.read(io.BytesIO(content), dtype='float32', always_2d=True)
        if sr < 1000 or sr > 192000 or audio.size == 0:
            raise ValueError('Invalid audio sample rate or empty file')
        start = time.perf_counter()
        x = features(torch.from_numpy(audio.T.copy()), sr).unsqueeze(0)
        with torch.no_grad():
            scores = torch.softmax(state['model'](x), dim=1)[0]
        latency = (time.perf_counter() - start) * 1000
    except Exception as exc:
        raise HTTPException(400, f'Invalid audio: {exc}') from exc
    probs = {name: round(float(scores[i]), 6) for i, name in enumerate(state['labels'])}
    index = int(scores.argmax())
    return {'label': state['labels'][index], 'confidence': round(float(scores[index]), 6), 'probabilities': probs, 'latency_ms': round(latency, 3), 'model_version': state['metrics']['model_version']}

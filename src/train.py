import argparse
import json
import random
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import Commands
from .model import SpeechCNN

URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'

def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'validation_list.txt').exists() and (root / 'testing_list.txt').exists():
        return root
    archive = root.parent / 'speech_commands_v0.02.tar.gz'
    if not archive.exists():
        print('Downloading Google Speech Commands (~2.4 GB)...', flush=True)
        urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive, 'r:gz') as tar:
        # Reject symlinks and paths escaping extraction root.
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if not target.is_relative_to(root.resolve()) or not member.isfile() and not member.isdir():
                raise ValueError('Unsafe dataset archive member')
        tar.extractall(root)
    return root

def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return 100.0 * correct / max(total, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='+', default=['yes', 'no', 'up', 'down'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data', default='data/SpeechCommands')
    parser.add_argument('--out', default='artifacts')
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    root = prepare(args.data)
    train = DataLoader(Commands(root, args.classes, 'training'), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val = DataLoader(Commands(root, args.classes, 'validation'), batch_size=args.batch_size, num_workers=0)
    test = DataLoader(Commands(root, args.classes, 'testing'), batch_size=args.batch_size, num_workers=0)
    model = SpeechCNN(len(args.classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(args.epochs):
        model.train()
        for x, y in train:
            optimizer.zero_grad(); loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward(); optimizer.step()
        print(f'Epoch {epoch + 1}/{args.epochs}, validation accuracy: {accuracy(model, val):.2f}%', flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'model.pt')
    (out / 'labels.json').write_text(json.dumps(args.classes))
    metrics = {'accuracy': round(accuracy(model, test), 3), 'model_version': '1.0.0', 'architecture': 'SpeechCNN', 'epochs': args.epochs, 'seed': args.seed}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    sample_dir = Path('samples'); sample_dir.mkdir(exist_ok=True)
    for label in args.classes:
        source = next((root / label).glob('*.wav'))
        import shutil
        shutil.copy2(source, sample_dir / f'{label}.wav')
    print(metrics)

if __name__ == '__main__':
    main()

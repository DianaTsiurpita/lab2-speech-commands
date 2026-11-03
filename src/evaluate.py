import argparse
import json
import torch
from torch.utils.data import DataLoader
from .data import Commands
from .model import SpeechCNN
from .train import accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/SpeechCommands')
    parser.add_argument('--artifacts', default='artifacts')
    args = parser.parse_args()
    from pathlib import Path
    path = Path(args.artifacts)
    labels = json.loads((path / 'labels.json').read_text())
    model = SpeechCNN(len(labels))
    model.load_state_dict(torch.load(path / 'model.pt', map_location='cpu', weights_only=True))
    loader = DataLoader(Commands(args.data, labels, 'testing'), batch_size=64)
    print(f'Accuracy: {accuracy(model, loader):.2f}%')

if __name__ == '__main__':
    main()

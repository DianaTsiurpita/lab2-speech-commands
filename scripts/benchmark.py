import argparse
import json
import statistics
import time
from pathlib import Path
import requests

def stats(values):
    values = sorted(values)
    return {'mean': round(statistics.mean(values), 3), 'p50': round(statistics.median(values), 3), 'p95': round(values[min(len(values)-1, int(0.95 * len(values)) - 1)], 3)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8000')
    parser.add_argument('--sample', default='samples/yes.wav')
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output', default='benchmark.json')
    args = parser.parse_args()
    payload = Path(args.sample).read_bytes()
    server, client = [], []
    for i in range(args.warmup + args.requests):
        start = time.perf_counter()
        response = requests.post(args.url + '/predict', files={'file': ('sample.wav', payload, 'audio/wav')}, timeout=60)
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        if i >= args.warmup:
            client.append(elapsed)
            server.append(response.json()['latency_ms'])
    result = {'server_ms': stats(server), 'client_ms': stats(client), 'requests': args.requests, 'warmup': args.warmup}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

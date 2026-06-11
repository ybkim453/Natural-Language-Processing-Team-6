'''
Post-Training Quantization (PTQ): FP32 → INT8.

이미 학습된 sst-classifier.pt를 로드하여 dynamic quantization을 적용하고,
정확도 손실과 모델 크기 변화를 측정한다.

실행:
  python ptq.py
'''

import io
import json
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classifier import GPT2SentimentClassifier, SentimentDataset, load_data, model_eval

CHECKPOINT = 'sst-classifier.pt'
BATCH_SIZE = 32
RESULTS_DIR = 'predictions'


def quantize_model(model: nn.Module) -> nn.Module:
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )


def model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024 / 1024


def measure_latency_ms(model: nn.Module, dataloader: DataLoader,
                       device: torch.device, n_batches: int = 20) -> float:
    model.eval()
    times = []
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        b_ids = batch['token_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        start = time.perf_counter()
        with torch.no_grad():
            model(b_ids, b_mask)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times) * 1000


def main():
    # dynamic quantization은 CPU에서만 동작
    device = torch.device('cpu')

    saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    config = saved['model_config']

    model_fp32 = GPT2SentimentClassifier(config).to(device)
    model_fp32.load_state_dict(saved['model'])
    model_fp32.eval()

    model_int8 = quantize_model(model_fp32)

    dev_data = load_data('data/ids-sst-dev.csv', 'valid')
    dev_dataset = SentimentDataset(dev_data, saved['args'])
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=BATCH_SIZE,
                            collate_fn=dev_dataset.collate_fn)

    print("FP32 평가 중...")
    fp32_acc, fp32_f1, *_ = model_eval(dev_loader, model_fp32, device)
    fp32_size = model_size_mb(model_fp32)
    fp32_latency = measure_latency_ms(model_fp32, dev_loader, device)

    print("INT8 평가 중...")
    int8_acc, int8_f1, *_ = model_eval(dev_loader, model_int8, device)
    int8_size = model_size_mb(model_int8)
    int8_latency = measure_latency_ms(model_int8, dev_loader, device)

    print(f"\n{'='*60}")
    print(f"{'Post-Training Quantization 결과':^60}")
    print(f"{'='*60}")
    print(f"{'모델':<20} {'Acc':>8} {'F1':>8} {'크기(MB)':>10} {'추론(ms)':>10}")
    print(f"{'-'*60}")
    print(f"{'FP32':<20} {fp32_acc:>8.3f} {fp32_f1:>8.3f} {fp32_size:>10.1f} {fp32_latency:>10.1f}")
    print(f"{'INT8 (quantized)':<20} {int8_acc:>8.3f} {int8_f1:>8.3f} {int8_size:>10.1f} {int8_latency:>10.1f}")
    print(f"{'='*60}")
    print(f"정확도 손실:  {fp32_acc - int8_acc:+.3f}")
    print(f"크기 감소:    {(1 - int8_size/fp32_size)*100:.1f}%")
    print(f"속도 변화:    {(fp32_latency/int8_latency - 1)*100:+.1f}%")

    # 결과 JSON 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        'method': 'ptq',
        'checkpoint': CHECKPOINT,
        'fp32': {
            'acc': round(fp32_acc, 4), 'f1': round(fp32_f1, 4),
            'size_mb': round(fp32_size, 2), 'latency_ms': round(fp32_latency, 2),
        },
        'int8': {
            'acc': round(int8_acc, 4), 'f1': round(int8_f1, 4),
            'size_mb': round(int8_size, 2), 'latency_ms': round(int8_latency, 2),
        },
        'accuracy_drop': round(fp32_acc - int8_acc, 4),
        'size_reduction_pct': round((1 - int8_size / fp32_size) * 100, 2),
        'speedup_pct': round((fp32_latency / int8_latency - 1) * 100, 2),
    }
    result_path = os.path.join(RESULTS_DIR, 'ptq_result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {result_path}")


if __name__ == '__main__':
    main()

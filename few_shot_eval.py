import csv
import json
import os
import random
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

CANDIDATES = [' 0', ' 1', ' 2', ' 3', ' 4']

# 레이블별 수작업 예시 — 클래스당 5개 (k-shot 최대 5까지 지원)
EXAMPLES_BY_LABEL = {
    0: [
        "a complete waste of time and money .",
        "this is one of the worst films i have ever seen .",
        "made me unintentionally famous -- as the queasy-stomached critic who staggered from the theater and blacked out in the lobby .",
        "a party-hearty teen flick that scalds like acid .",
        "dull , lifeless , and completely devoid of any entertainment value .",
    ],
    1: [
        "it 's not a great monster movie .",
        "this is n't a new idea .",
        "frida is n't that much different from many a hollywood romance .",
        "the film fails to deliver on its promise .",
        "disappointing and forgettable .",
    ],
    2: [
        "the film is neither good nor bad .",
        "a passable but unremarkable entry in the genre .",
        "it has its moments but ultimately falls short .",
        "an average film that does nothing particularly wrong or right .",
        "watchable but not especially memorable .",
    ],
    3: [
        "yet the act is still charming here .",
        "`` extreme ops '' exceeds expectations .",
        "it 's been done before but never so vividly or with so much passion .",
        "a solid and entertaining film .",
        "worth watching for its strong performances .",
    ],
    4: [
        "the actors are fantastic .",
        "the gorgeously elaborate continuation of `` the lord of the rings '' trilogy is so huge that a column of words can not adequately describe co-writer\/director peter jackson 's expanded vision of j.r.r. tolkien 's middle-earth .",
        "a masterpiece of modern cinema .",
        "an unforgettable and deeply moving experience .",
        "one of the best films of the year .",
    ],
}


def seed_everything(seed=11711):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_sst(filename):
    data = []
    with open(filename, 'r') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            sent = record['sentence'].lower().strip()
            label = int(record['sentiment'].strip())
            data.append((sent, label))
    return data


def build_prompt(test_sent, examples):
    prompt = ''
    for sent, label in examples:
        prompt += f'Review: "{sent}"\nSentiment: {label}\n\n'
    prompt += f'Review: "{test_sent}"\nSentiment:'
    return prompt


def get_prediction(model, tokenizer, prompt, device):
    results = {}
    for candidate in CANDIDATES:
        full_text = prompt + candidate
        inputs = tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=950
        ).to(device)

        prompt_inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=950
        ).to(device)

        prompt_len = prompt_inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            candidate_ids = inputs['input_ids'][0][prompt_len:]

            log_prob = 0.0
            for i, token_id in enumerate(candidate_ids):
                pos = prompt_len - 1 + i
                log_prob += torch.log_softmax(logits[pos], dim=-1)[token_id].item()

        results[candidate.strip()] = log_prob

    return int(max(results, key=results.get))


def get_examples_for_k(k):
    """클래스당 k개씩, 총 k*5개 예시 반환."""
    if k == 0:
        return []
    examples = []
    for label, sents in EXAMPLES_BY_LABEL.items():
        for sent in sents[:k]:
            examples.append((sent, label))
    random.shuffle(examples)
    return examples


def evaluate(model, tokenizer, test_data, k, device):
    examples = get_examples_for_k(k)
    correct = 0
    total = 0
    records = []

    for sent, label in tqdm(test_data, desc=f'{k}-shot ({k*5} examples)'):
        prompt = build_prompt(sent, examples)
        pred = get_prediction(model, tokenizer, prompt, device)

        if pred == label:
            correct += 1
        total += 1
        records.append((sent, label, pred))

    acc = correct / total if total > 0 else 0.0

    out_path = os.path.join('predictions', f'few-shot-{k}shot-dev-out.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sentence', 'true_label', 'predicted_label'])
        writer.writerows(records)

    return acc


def load_finetuned_results():
    results = {}
    for mode in ['last-linear-layer', 'full-model']:
        path = os.path.join('predictions', f'{mode}-sst_result.json')
        with open(path) as f:
            results[mode] = json.load(f)['dev_acc']
    return results


def main():
    seed_everything(11711)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('Loading GPT-2 base model...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()

    print('Loading SST data...')
    dev_data = load_sst('data/ids-sst-dev.csv')
    print(f'eval samples: {len(dev_data)}')

    finetuned_results = load_finetuned_results()

    print('\n===== Few-shot vs Zero-shot 평가 결과 (SST 5-class, 클래스당 k개) =====\n')
    results = {}

    for k in [0, 1, 3, 5]:
        n_examples = k * 5
        print(f'\n[{k}-shot] 프롬프트 예시 {n_examples}개 사용')
        acc = evaluate(model, tokenizer, dev_data, k, device)
        results[k] = acc
        print(f'{k}-shot accuracy: {acc:.3f}')

    print('\n----- 최종 비교표 -----')
    print(f'{"방법":<40} {"예시 수":>8} {"Accuracy":>10}')
    print('-' * 60)
    for k, acc in results.items():
        print(f'{k}-shot (base GPT-2){"":<22} {k*5:>8} {acc:>10.3f}')
    print('-' * 60)
    for method, acc in finetuned_results.items():
        print(f'fine-tuned ({method}){"":<18} {"전체":>8} {acc:>10.3f}')

    result = {
        'method': 'few_shot',
        'task': 'sst-5class',
        'scheme': 'k-shot per class',
        'few_shot_results': {f'{k}-shot': round(acc, 4) for k, acc in results.items()},
        'finetuned_baseline': finetuned_results,
    }
    result_path = os.path.join('predictions', 'few_shot_result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {result_path}")

    print('\n완료!')


if __name__ == '__main__':
    main()

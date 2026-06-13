import csv
import random
import os
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

LABEL_MAP = {
    0: 'negative',
    1: 'negative',
    2: 'neutral',
    3: 'positive',
    4: 'positive',
}

VALID_LABELS = {0, 1, 3, 4}

MANUAL_POS = [
    "yet the act is still charming here .",
    "the actors are fantastic .",
    "`` extreme ops '' exceeds expectations .",
    "it 's been done before but never so vividly or with so much passion .",
    "the gorgeously elaborate continuation of `` the lord of the rings '' trilogy is so huge that a column of words can not adequately describe co-writer\/director peter jackson 's expanded vision of j.r.r. tolkien 's middle-earth .",
]

MANUAL_NEG = [
    "this is n't a new idea .",
    "it 's not a great monster movie .",
    "a party-hearty teen flick that scalds like acid .",
    "frida is n't that much different from many a hollywood romance .",
    "made me unintentionally famous -- as the queasy-stomached critic who staggered from the theater and blacked out in the lobby .",
]


def seed_everything(seed=11711):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_sst(filename):
    data = []
    with open(filename, 'r') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            sent = record['sentence'].lower().strip()
            label = int(record['sentiment'].strip())
            if label in VALID_LABELS:
                data.append((sent, label))
    return data


def build_prompt(test_sent, examples):
    prompt = ''
    for sent, sentiment in examples:
        prompt += f'Review: "{sent}"\nSentiment: {sentiment}\n\n'
    prompt += f'Review: "{test_sent}"\nSentiment:'
    return prompt


def get_prediction(model, tokenizer, prompt, device):
    results = {}
    for candidate in [' positive', ' negative']:
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

    return 'positive' if results['positive'] > results['negative'] else 'negative'


def get_examples_for_k(k):
    if k == 0:
        return []
    half = k // 2
    remainder = k % 2
    pos = [(s, 'positive') for s in MANUAL_POS[:half + remainder]]
    neg = [(s, 'negative') for s in MANUAL_NEG[:half]]
    examples = pos + neg
    random.shuffle(examples)
    return examples


def evaluate(model, tokenizer, test_data, k, device):
    examples = get_examples_for_k(k)
    correct = 0
    total = 0
    predictions = []

    for sent, label in tqdm(test_data, desc=f'{k}-shot'):
        true_sentiment = LABEL_MAP[label]
        prompt = build_prompt(sent, examples)
        pred = get_prediction(model, tokenizer, prompt, device)
        predictions.append((sent, true_sentiment, pred))

        if pred == true_sentiment:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return acc, predictions


def save_predictions(predictions, k, acc):
    os.makedirs('predictions', exist_ok=True)
    filepath = f'predictions/fewshot-{k}shot-sst-dev-out.csv'
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['sentence', 'true_label', 'predicted_label'])
        for sent, true, pred in predictions:
            writer.writerow([sent, true, pred])
    print(f'saved: {filepath} (acc: {acc:.3f})')


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
    eval_data = dev_data
    print(f'eval samples: {len(eval_data)}')

    finetuned_results = {
        'last-linear-layer': 0.461,
        'full-model': 0.407,
    }

    print('\n===== Few-shot vs Zero-shot 평가 결과 =====\n')
    results = {}

    for k in [0, 1, 3, 5]:
        acc, preds = evaluate(model, tokenizer, eval_data, k, device)
        results[k] = acc
        save_predictions(preds, k, acc)
        print(f'{k}-shot accuracy: {acc:.3f}')

    print('\n----- 최종 비교표 -----')
    print(f'{"방법":<35} {"Accuracy":>10}')
    print('-' * 47)
    for k, acc in results.items():
        print(f'{k}-shot (base GPT-2){"":<17} {acc:>10.3f}')
    print('-' * 47)
    for method, acc in finetuned_results.items():
        print(f'fine-tuned ({method}){"":<13} {acc:>10.3f}')

    print('\n완료!')

if __name__ == '__main__':
    main()
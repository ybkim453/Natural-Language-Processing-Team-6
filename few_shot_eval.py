
import random
import csv
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

# SST label: 0=매우부정, 1=부정, 2=중립, 3=긍정, 4=매우긍정
LABEL_MAP = {
    0: 'negative',
    1: 'negative',
    2: 'neutral',
    3: 'positive',
    4: 'positive',
}

VALID_LABELS = {0, 1, 3, 4}  # 중립 제외


def seed_everything(seed=11711):
    random.seed(seed)
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
    '''k-shot 프롬프트 생성'''
    prompt = ''
    for sent, label in examples:
        sentiment = LABEL_MAP[label]
        prompt += f'Review: "{sent}"\nSentiment: {sentiment}\n\n'
    prompt += f'Review: "{test_sent}"\nSentiment:'
    return prompt


def get_prediction(model, tokenizer, prompt, device):
    '''
    positive vs negative 확률 비교
    - 단순 토큰 확률이 아닌 log-likelihood 방식으로 bias 줄임
    - " positive"와 " negative" 각각의 조건부 확률 비교
    '''
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
            outputs = model(**inputs, labels=inputs['input_ids'])
            # candidate 부분의 log-likelihood만 계산
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            candidate_ids = inputs['input_ids'][0][prompt_len:]

            log_prob = 0.0
            for i, token_id in enumerate(candidate_ids):
                pos = prompt_len - 1 + i
                log_prob += torch.log_softmax(logits[pos], dim=-1)[token_id].item()

        results[candidate.strip()] = log_prob

    return 'positive' if results['positive'] > results['negative'] else 'negative'


def evaluate(model, tokenizer, test_data, train_pos_pool, train_neg_pool, k, device):
    '''
    k-shot accuracy 측정
    - test_data: dev set (한 번만 사용)
    - 예시는 train_pos_pool, train_neg_pool에서만 추출
    '''
    correct = 0
    total = 0

    for sent, label in tqdm(test_data, desc=f'{k}-shot'):
        true_sentiment = LABEL_MAP[label]

        # train에서 균형 맞춰 예시 추출
        if k > 0:
            half = k // 2
            remainder = k % 2
            examples = (
                random.sample(train_pos_pool, half + remainder) +
                random.sample(train_neg_pool, half)
            )
            random.shuffle(examples)
        else:
            examples = []

        prompt = build_prompt(sent, examples)
        pred = get_prediction(model, tokenizer, prompt, device)

        if pred == true_sentiment:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def main():
    seed_everything(11711)
    device = torch.device('cpu')

    print('Loading GPT-2 base model...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()

    print('Loading SST data...')
    train_data = load_sst('data/ids-sst-train.csv')
    dev_data = load_sst('data/ids-sst-dev.csv')

    # 평가용 dev set 300개 고정 (딱 한 번만 사용)
    eval_data = dev_data[:300]

    # 예시용 train pool (positive/negative 분리)
    train_pos_pool = [(s, l) for s, l in train_data if LABEL_MAP[l] == 'positive']
    train_neg_pool = [(s, l) for s, l in train_data if LABEL_MAP[l] == 'negative']
    print(f'train pos: {len(train_pos_pool)}, train neg: {len(train_neg_pool)}')
    print(f'eval samples: {len(eval_data)}')

    # fine-tuned 결과 (이미 돌린 결과)
    finetuned_results = {
        'last-linear-layer': 0.461,
    }

    print('\n===== Few-shot vs Zero-shot 평가 결과 =====\n')
    results = {}

    for k in [0, 1, 3, 5]:
        acc = evaluate(model, tokenizer, eval_data, train_pos_pool, train_neg_pool, k, device)
        results[k] = acc
        print(f'{k}-shot accuracy: {acc:.3f}')

    print('\n----- 비교표 -----')
    print(f'{"방법":<35} {"Accuracy":>10}')
    print('-' * 47)
    for k, acc in results.items():
        print(f'{k}-shot (base GPT-2){"":<17} {acc:>10.3f}')
    print('-' * 47)
    for method, acc in finetuned_results.items():
        print(f'fine-tuned ({method})       {acc:>10.3f}')

    print('\n완료!')


if __name__ == '__main__':
    main()
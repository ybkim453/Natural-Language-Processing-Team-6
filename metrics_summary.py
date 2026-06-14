import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


SENTIMENT_DATASETS = {
  'sst': 'ids-sst-dev.csv',
  'cfimdb': 'ids-cfimdb-dev.csv',
}


def round_metric(value):
  return round(float(value), 4)


def load_json(path, default=None):
  path = Path(path)
  if not path.exists():
    return {} if default is None else default
  with open(path, encoding='utf-8') as f:
    return json.load(f)


def write_json(path, data):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)


def read_gold_labels(path, label_column):
  labels = {}
  with open(path, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
      sent_id = row['id'].strip().lower()
      labels[sent_id] = int(row[label_column].strip())
  return labels


def read_prediction_labels(path):
  predictions = {}
  with open(path, encoding='utf-8') as f:
    for line in f:
      if ',' not in line:
        continue
      sent_id, pred = line.split(',', 1)
      sent_id = sent_id.strip().lower()
      pred = pred.strip()
      if not sent_id or sent_id == 'id':
        continue
      predictions[sent_id] = int(pred)
  return predictions


def macro_f1(y_true, y_pred):
  labels = sorted(set(y_true) | set(y_pred))
  if not labels:
    return 0.0

  f1_scores = []
  for label in labels:
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

  return sum(f1_scores) / len(f1_scores)


def classification_metrics(gold_path, prediction_path, label_column='sentiment'):
  gold = read_gold_labels(gold_path, label_column)
  predictions = read_prediction_labels(prediction_path)
  common_ids = [sent_id for sent_id in gold if sent_id in predictions]

  if not common_ids:
    raise ValueError(f'No overlapping ids between {gold_path} and {prediction_path}')

  y_true = [gold[sent_id] for sent_id in common_ids]
  y_pred = [predictions[sent_id] for sent_id in common_ids]
  correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

  return {
    'dev_acc': round_metric(correct / len(common_ids)),
    'dev_f1': round_metric(macro_f1(y_true, y_pred)),
    'num_examples': len(common_ids),
  }


def sentiment_result_path(predictions_dir, fine_tune_mode, dataset):
  return Path(predictions_dir) / f'{fine_tune_mode}-{dataset}_result.json'


def collect_sentiment_results(data_dir, predictions_dir):
  data_dir = Path(data_dir)
  predictions_dir = Path(predictions_dir)
  results = {}

  for dataset, gold_filename in SENTIMENT_DATASETS.items():
    for prediction_path in sorted(predictions_dir.glob(f'*-{dataset}-dev-out.csv')):
      suffix = f'-{dataset}-dev-out.csv'
      fine_tune_mode = prediction_path.name[:-len(suffix)]
      metrics = classification_metrics(data_dir / gold_filename, prediction_path)
      result = {
        'method': 'sentiment_analysis',
        'fine_tune_mode': fine_tune_mode,
        'dataset': dataset,
        **metrics,
      }
      results.setdefault(fine_tune_mode, {})[dataset] = result

  return results


def selected_sentiment_mode(sentiment_results):
  if 'full-model' in sentiment_results:
    return 'full-model'
  if not sentiment_results:
    return None
  return sorted(sentiment_results)[0]


def split_sonnets(path):
  text = Path(path).read_text(encoding='utf-8')
  return [sonnet.strip() for sonnet in re.split(r'\n\s*\d+\s*\n', text)[1:] if sonnet.strip()]


def char_ngrams(text, order):
  text = ''.join(text.split())
  return Counter(text[i:i + order] for i in range(max(len(text) - order + 1, 0)))


def fallback_chrf(generated_sonnets, true_sonnets, char_order=6, beta=2.0):
  max_len = min(len(generated_sonnets), len(true_sonnets))
  if max_len == 0:
    return 0.0

  generated_sonnets = generated_sonnets[:max_len]
  true_sonnets = true_sonnets[:max_len]
  precisions = []
  recalls = []

  for order in range(1, char_order + 1):
    matches = 0
    generated_total = 0
    true_total = 0

    for generated, true in zip(generated_sonnets, true_sonnets):
      generated_counts = char_ngrams(generated, order)
      true_counts = char_ngrams(true, order)
      matches += sum((generated_counts & true_counts).values())
      generated_total += sum(generated_counts.values())
      true_total += sum(true_counts.values())

    if generated_total or true_total:
      precisions.append(matches / generated_total if generated_total else 0.0)
      recalls.append(matches / true_total if true_total else 0.0)

  if not precisions or not recalls:
    return 0.0
  precision = sum(precisions) / len(precisions)
  recall = sum(recalls) / len(recalls)
  if precision == 0.0 and recall == 0.0:
    return 0.0

  beta_squared = beta ** 2
  return 100.0 * (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def compute_chrf(generated_path, gold_path):
  generated_sonnets = split_sonnets(generated_path)
  true_sonnets = split_sonnets(gold_path)
  max_len = min(len(generated_sonnets), len(true_sonnets))
  generated_sonnets = generated_sonnets[:max_len]
  true_sonnets = true_sonnets[:max_len]

  try:
    from sacrebleu.metrics import CHRF

    score = CHRF().corpus_score(generated_sonnets, [true_sonnets]).score
  except ImportError:
    score = fallback_chrf(generated_sonnets, true_sonnets)

  return round_metric(score)


def default_sonnet_gold_path(data_dir):
  data_dir = Path(data_dir)
  candidates = [
    data_dir / 'TRUE_sonnets_held_out.txt',
    data_dir / 'TRUE_sonnets_held_out_dev.txt',
  ]
  for candidate in candidates:
    if candidate.exists():
      return candidate
  return None


def default_generated_sonnet_path(predictions_dir):
  predictions_dir = Path(predictions_dir)
  candidates = [
    predictions_dir / 'generated_sonnets_dev.txt',
    predictions_dir / 'generated_sonnets.txt',
  ]
  for candidate in candidates:
    if candidate.exists():
      return candidate
  return None


def collect_sonnet_results(data_dir, predictions_dir):
  data_dir = Path(data_dir)
  predictions_dir = Path(predictions_dir)
  result_path = predictions_dir / 'sonnet_result.json'
  result = load_json(result_path)

  best_dev_loss = result.get('best_dev_loss')
  if best_dev_loss is not None:
    result['perplexity'] = round_metric(math.exp(float(best_dev_loss)))

  generated_path = default_generated_sonnet_path(predictions_dir)
  gold_path = default_sonnet_gold_path(data_dir)
  if generated_path is not None and gold_path is not None:
    result['chrf'] = compute_chrf(generated_path, gold_path)
    result['chrf_generated_path'] = str(generated_path)
    result['chrf_gold_path'] = str(gold_path)
    if generated_path.name == 'generated_sonnets.txt' and gold_path.name.endswith('_dev.txt'):
      result['chrf_warning'] = (
        'generated_sonnets_dev.txt was not found; CHRF used generated_sonnets.txt '
        'with the available dev gold file.'
      )
    else:
      result.pop('chrf_warning', None)

  result.setdefault('method', 'sonnet_generation')
  return result


def build_metrics_summary(data_dir='data', predictions_dir='predictions'):
  data_dir = Path(data_dir)
  predictions_dir = Path(predictions_dir)

  sentiment_results = collect_sentiment_results(data_dir, predictions_dir)
  mode = selected_sentiment_mode(sentiment_results)
  selected_sentiment = sentiment_results.get(mode, {}) if mode is not None else {}
  paraphrase_result = load_json(predictions_dir / 'para_result.json')
  sonnet_result = collect_sonnet_results(data_dir, predictions_dir)

  return {
    'sentiment_analysis': {
      'selected_mode': mode,
      'sst_acc': selected_sentiment.get('sst', {}).get('dev_acc'),
      'cfimdb_acc': selected_sentiment.get('cfimdb', {}).get('dev_acc'),
      'all_results': sentiment_results,
    },
    'paraphrase_detection': {
      'acc': paraphrase_result.get('dev_acc'),
      'f1': paraphrase_result.get('dev_f1'),
      'source': str(predictions_dir / 'para_result.json'),
    },
    'sonnet_generation': {
      'chrf': sonnet_result.get('chrf'),
      'perplexity': sonnet_result.get('perplexity'),
      'best_dev_loss': sonnet_result.get('best_dev_loss'),
      'source': str(predictions_dir / 'sonnet_result.json'),
    },
    'table': {
      'sst_acc': selected_sentiment.get('sst', {}).get('dev_acc'),
      'cfimdb_acc': selected_sentiment.get('cfimdb', {}).get('dev_acc'),
      'paraphrase_acc': paraphrase_result.get('dev_acc'),
      'sonnet_chrf': sonnet_result.get('chrf'),
      'sonnet_perplexity': sonnet_result.get('perplexity'),
    },
  }


def format_metric(value):
  return '' if value is None else str(value)


def format_metrics_table(summary):
  table = summary['table']
  return '\n'.join([
    '| SST Acc. | CFIMDB Acc. | Paraphrase Acc. | CHRF | Perplexity |',
    '| ---: | ---: | ---: | ---: | ---: |',
    (
      f"| {format_metric(table['sst_acc'])} | {format_metric(table['cfimdb_acc'])} | "
      f"{format_metric(table['paraphrase_acc'])} | {format_metric(table['sonnet_chrf'])} | "
      f"{format_metric(table['sonnet_perplexity'])} |"
    ),
    '',
  ])


def write_all_metrics(data_dir='data', predictions_dir='predictions'):
  predictions_dir = Path(predictions_dir)
  predictions_dir.mkdir(parents=True, exist_ok=True)

  sentiment_results = collect_sentiment_results(data_dir, predictions_dir)
  for fine_tune_mode, dataset_results in sentiment_results.items():
    for dataset, result in dataset_results.items():
      write_json(sentiment_result_path(predictions_dir, fine_tune_mode, dataset), result)

  sonnet_result = collect_sonnet_results(data_dir, predictions_dir)
  write_json(predictions_dir / 'sonnet_result.json', sonnet_result)

  summary = build_metrics_summary(data_dir, predictions_dir)
  write_json(predictions_dir / 'final_metrics.json', summary)
  (predictions_dir / 'final_metrics.md').write_text(format_metrics_table(summary), encoding='utf-8')

  return summary


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default='data')
  parser.add_argument('--predictions_dir', default='predictions')
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  summary = write_all_metrics(args.data_dir, args.predictions_dir)
  print(format_metrics_table(summary))

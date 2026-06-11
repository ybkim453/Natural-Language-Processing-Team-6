# Natural-Language-Processing-Team-6

## 환경 설정

```bash
conda env create -f env.yml
conda activate nlp_proj
```

---

## PART I: GPT-2 핵심 구현

### 구현 파일
| 파일 | 내용 |
|------|------|
| `modules/attention.py` | CausalSelfAttention |
| `modules/gpt2_layer.py` | GPT2Layer |
| `models/gpt2.py` | GPT2Model |
| `optimizer.py` | AdamW |
| `classifier.py` | GPT2SentimentClassifier |

### 실행

**옵티마이저 테스트**
```bash
python optimizer_test.py
```

**GPT-2 모델 sanity check**
```bash
python sanity_check.py
```

**감정 분류 (SST / CFIMDB)**
```bash
python classifier.py --fine-tune-mode last-linear-layer --use_gpu
python classifier.py --fine-tune-mode full-model --use_gpu
```

---

## PART II: 태스크 확장

### 구현 파일
| 파일 | 내용 |
|------|------|
| `paraphrase_detection.py` | ParaphraseGPT — Quora 패러프레이즈 탐지 |
| `sonnet_generation.py` | SonnetGPT — 소넷 생성 (temperature / top-k / top-p) |
| `few_shot_eval.py` | Few-shot vs Zero-shot SST 정확도 비교 |

### 실행

**패러프레이즈 탐지**
```bash
python paraphrase_detection.py --use_gpu
```

**소넷 생성**
```bash
python sonnet_generation.py --use_gpu
```

**Few-shot vs Zero-shot 평가**
```bash
python few_shot_eval.py
```

---

## PART III: 파라미터 효율화 및 모델 압축 (Extra)

### 구현 파일
| 파일 | 내용 |
|------|------|
| `lora.py` | LoRA — attention Q/V에 low-rank 어댑터 삽입 |
| `prefix_tuning.py` | Prefix Tuning — attention 레이어에 학습 가능한 prefix K/V 삽입 |
| `ptq.py` | Post-Training Quantization — FP32 → INT8 변환 |

### 실행

**LoRA** (rank별 비교)
```bash
python lora.py --use_gpu --lora_r 4
python lora.py --use_gpu --lora_r 8
python lora.py --use_gpu --lora_r 16
```

**Prefix Tuning** (prefix 길이별 비교)
```bash
python prefix_tuning.py --use_gpu --prefix_length 10
python prefix_tuning.py --use_gpu --prefix_length 20
python prefix_tuning.py --use_gpu --prefix_length 50
```

**Post-Training Quantization**
```bash
python ptq.py
```
> `sst-classifier.pt` 가 필요합니다 (`classifier.py` 먼저 실행).

### 결과물
모든 실행 결과는 `predictions/` 폴더에 저장됩니다.
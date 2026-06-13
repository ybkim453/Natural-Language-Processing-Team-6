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

## LoRA 실험

`CausalSelfAttention`의 query/value projection에 LoRA 어댑터를 삽입했다. `lora` 모드에서는 GPT-2 backbone 전체를 고정하고, 각 attention layer의 Q/V LoRA 파라미터만 학습한다.

Paraphrase Detection accuracy 실험:

```bash
conda run -n nlp_proj python paraphrase_detection.py --fine_tune_mode lora --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 --use_gpu
conda run -n nlp_proj python paraphrase_detection.py --fine_tune_mode full-model --use_gpu
```

Sonnet Generation 실험:

```bash
conda run -n nlp_proj python sonnet_generation.py --fine_tune_mode lora --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 --use_gpu
conda run -n nlp_proj python sonnet_generation.py --fine_tune_mode full-model --use_gpu
```

기본 GPT-2 설정(`gpt2`, rank=8, Q/V LoRA)의 학습 파라미터 수 비교:

| Mode | Trainable params | Total params | Trainable % |
| --- | ---: | ---: | ---: |
| full-model | 110,865,408 | 110,865,408 | 100.0000% |
| lora | 294,912 | 111,160,320 | 0.2653% |

학습 로그에는 동일한 형식의 파라미터 비교표가 출력된다. Paraphrase는 `Best dev acc (lora)` 및 `dev paraphrase acc`를 LoRA 적용 모델 accuracy로 기록하고, full-model 실행 결과와 같은 epoch/lr 조건에서 비교한다.
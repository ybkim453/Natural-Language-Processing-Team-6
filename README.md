# Natural-Language-Processing-Team-6

## PART-I:

#### 다음 각 모듈에서 누락된 코드 블록을 완성해야 한다.
* `modules/attention.py`
* `modules/gpt2_layer.py`
* `models/gpt2.py`
* `classifier.py`
* `optimizer.py`

#### 다음 모듈들을 실행하여 PART-I의 구현을 테스트한다.

* `optimizer_test.py`: `optimizer.py` 구현을 테스트.
* `sanity_check.py`: GPT 모델 구현을 테스트.
* `classifier.py`: 모델을 사용한 감정 분류 수행.

## PART-II

#### 다음 모듈들을 실행하여 PART-II의 구현을 테스트한다.

* `paraphrase_detection.py`: 패러프레이즈 탐지 수행.
* `sonnet_generation.py`: 소네트 생성 수행.

**주목**: 사용하는 GPU 사양에 따라 batch_size 같은 하이퍼파라미터를 조정하여 성능을 최적화하고 메모리 부족 오류를 방지해야 한다.

#### PART-II 테스트의 핵심 포인트

두 파일에 있는 누락된 코드 블록을 완성하는 것도 중요하지만, PART-II의 핵심은 기능의 확장에 있다. GPT-2 모델을 수정하여 한 문장이 다른 문장의 패러프레이즈인지 판단하는 능력과 소네트를 생성하는 능력을 개선하는 방법에 촛점을 맞추도록 하자.

#### 파이썬 설치
* anaconda3 를 설치한다.

#### 환경 및 패키지 설치

* conda env create -f env.yml
* conda activate nlp_proj 

**주의**:
* 프로젝트 PART-I을 수행하면서, 위에서 설치된 패키지만을 사용해야 하며, 별도의 다른 패키지는 허용되지 않는다.
* 모든 command 옵션이나 파라미터는 변경/추가하면 안된다.

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


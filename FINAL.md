# Hairline Conditioned Diffusion v2 – Final Report

## 1. 프로젝트 개요와 목적
- **목표**: 의사가 설계한 머리선(hairline mask)에 맞춰 대머리 이미지를 자연스러운 머리 이미지로 복원하는 조건부 Stable Diffusion 모델을 구축한다.
- **핵심 아이디어**: Diffusion의 복원 목표(x₀)를 대머리 이미지가 아니라 원본 머리 이미지의 latent(`z_orig`)로 두고, 대머리 이미지 latent와 hairline mask를 조건으로만 사용한다.
- **기대 효과**: 임의의 hairline mask와 텍스트 프롬프트를 넣어도 현실적인 머리카락을 합성할 수 있으며, 의학적/미용적 시뮬레이션 워크플로우에 활용 가능하다.

## 2. 데이터 구성 및 전처리
| 구분 | 설명 | 경로 인자 |
| --- | --- | --- |
| 입력 이미지 `I_orig` | 머리가 있는 원본 얼굴 (정답) | `--orig_dir` |
| 조건 이미지 `I_bald` | 정렬된 대머리 얼굴 (조건) | `--bald_dir` |
| 헤어라인 마스크 `M` | 이마 구역 1채널 마스크 | `--mask_dir` |
| 텍스트 메타데이터 | 선택적 프롬프트(`prompt`) | `--metadata_path`, `--metadata_text_key` |

- 해상도는 기본 512×512이며, `HairlineDatasetV2`가 PIL 이미지를 로드해 정규화 후 Tensor로 반환한다.
- 마스크는 학습 시 UNet 입력용으로 64×64까지 다운샘플링되고, conditioner 토큰 생성에도 재사용된다.

## 3. 모델 구성
- **기반 모델**: `runwayml/stable-diffusion-v1-5`
  - **VAE**(`AutoencoderKL`): `I_orig`, `I_bald`를 latent 공간(4×64×64)으로 매핑.
  - **Text Encoder**(`CLIPTextModel`): 프롬프트를 임베딩으로 변환하고 freeze.
  - **UNet**(`UNet2DConditionModel`): `enable_hairline_conditioning`을 통해 입력 채널을 5(4 latent + 1 mask)로 확장.
- **HairlineConditioningEmbeddings**
  - `mask_latent`를 글로벌 평균 풀링 후 선형층으로 투사하여 `mask_token` 생성.
  - 옵션으로 `z_bald`를 풀링한 `bald_token`을 추가 (`--use_bald_token`).
  - 생성된 토큰을 CLIP 텍스트 토큰 앞에 붙여 cross-attention 입력을 구성.

## 4. Stable Diffusion v1.5 대비 변경점
- **UNet 입력 채널 확장**  
  - *무엇을*: 기본 4채널(latent) → 5채널(latent + mask)로 변경.  
  - *왜*: 공간적 hairline 정보를 직접 UNet에 주입해 생성된 머리카락이 의사의 마스크를 정확히 따르게 만들기 위함.  
  - *어떻게*: `enable_hairline_conditioning`이 UNet의 `in_channels`를 5로 재구성하고, 추론 시에도 동일하게 `torch.cat([latent, mask], dim=1)` 구조를 사용(`train_hairline_cond_v2.py:111`, `infer_hairline_cond_v2.py:141`).
- **Cross-Attention 토큰 커스터마이즈**  
  - *무엇을*: CLIP 텍스트 토큰 앞에 `mask_token`과 (옵션) `bald_token`을 삽입.  
  - *왜*: 텍스트만으로는 머리선 제약이 불완전하므로, 마스크/대머리 latent를 전역 스타일 조건으로 병합해 geometric & photometric 일관성을 확보.  
  - *어떻게*: `HairlineConditioningEmbeddings`가 풀링된 마스크/대머리 latent를 선형층에 통과시켜 cond 토큰을 만들고, 학습·추론 둘 다 `torch.cat([cond_tokens, text_tokens], dim=1)`로 전달.  
- **학습 목표 변경**  
  - *무엇을*: 일반 SD는 텍스트만 조건으로 삼지만, 본 모델은 `(mask, bald latent)`를 조건으로 두고 target `x₀`는 항상 `z_orig`.  
  - *왜*: 대머리 이미지를 target으로 학습하면 hairline 편집이 불가능하기 때문에, 원본 머리 latent를 복원해야 원하는 헤어스타일을 얻을 수 있음.  
  - *어떻게*: 학습 루프에서 `vae.encode(orig)`로 얻은 latent에 노이즈를 주입하고, `vae.encode(bald)`는 conditioner에만 공급(`train_hairline_cond_v2.py:94-150`).
- **추론 파이프라인 개선**  
  - *무엇을*: DDIM 기반 샘플러로 사용자 제공 마스크를 그대로 사용하고, classifier-free guidance를 지원.  
  - *왜*: 빠른 생성과 프롬프트 기반 제어를 동시에 만족시키고, 마스크를 1:1 반영하기 위함.  
  - *어떻게*: `infer_hairline_cond_v2.py`가 `DDIMScheduler`, guidance 스케일, 반복된 마스크 latent, conditioner 토큰을 구성해 한 번에 처리.

## 5. 입력/출력 및 조건 결합 구조
| 단계 | 입력 | 처리 | 출력 |
| --- | --- | --- | --- |
| Latent Encoding | `I_orig`, `I_bald`, `M` | VAE Encoder, 마스크 다운샘플링 | `z_orig`, `z_bald`, `m` |
| UNet 입력 | `z_orig`, `m` | 노이즈 주입(`x_t = add_noise(z_orig)`), concat | `[x_t, m]` (5채널) |
| Cond Token | `m`, `z_bald`, 텍스트 | Conditioner + CLIP | `[mask_token,(bald_token),text_tokens]` |
| 모델 출력 | `[x_t, m]`, cond, `t` | Hairline-conditioned UNet | 노이즈 추정치 `ε_pred` |
| 최종 복원 | `ε_pred` | Scheduler 역과정 + VAE Decoder | 머리카락 복원 이미지 |

## 6. 학습 목표 및 손실
- **훈련 목표**: `x₀ = z_orig`를 복원하도록 노이즈 예측 손실(MSE)을 최소화.
- **손실 식**:
  ```
  ε ~ N(0, I)
  x_t = √α_t · z_orig + √(1-α_t) · ε
  ε_pred = UNet([x_t, m], t, cond)
  L = MSE(ε, ε_pred)
  ```
- **Scheduler**: `DDPMScheduler`로 학습용 노이즈 및 역과정 파라미터를 관리.

## 7. 학습 파이프라인 (train_hairline_cond_v2.py)
- `Accelerate`로 다중 GPU/혼합 정밀 학습 및 gradient accumulation 지원.
- `AdamW` 옵티마이저 + `get_scheduler`로 LR 스케줄 (`cosine_with_restarts` 기본).
- 중요 하이퍼파라미터 (기본값):
  - 배치: `--train_batch_size 2`, 해상도 512, 에폭 `--num_train_epochs 1` 또는 `--max_train_steps`.
  - 학습률: `1e-5`, `--lr_warmup_steps 500`.
  - 클리핑: `--max_grad_norm 1.0`.
- 체크포인트: `--checkpointing_steps`마다 `accelerator.save_state`.
- 종료 시 산출물:
  - `output_dir/unet/` : 미세조정된 UNet 가중치.
  - `output_dir/conditioner.pt` : conditioner state dict 및 메타데이터(`hidden_size`, `use_bald_token`).
  - 토크나이저 복제본, 로그 디렉터리.

## 8. 추론 파이프라인 (infer_hairline_cond_v2.py)
| 구성 요소 | 설명 |
| --- | --- |
| 입력 | `--bald_path`, `--mask_path`, 선택적 `--prompt / --negative_prompt`, 샘플 수 `--num_samples`. |
| 전처리 | 이미지/마스크를 512로 리사이즈 후 정규화, 마스크는 64×64 latent로 보간. |
| 모델 로드 | 미세조정 UNet(`--model_dir`) + conditioner(`--conditioner_path`), 없으면 기본 모델에 hairline conditioning 활성화. |
| 샘플링 | `DDIMScheduler`(기본 스텝 30)와 classifier-free guidance(`--guidance_scale`)를 사용. |
| 출력 | VAE 디코더로 이미지를 복원 후 `out_dir/sample_###.png`로 저장. |

- Cond 토큰은 학습과 동일하게 구성되어 텍스트, 마스크,(및 대머리 latent) 정보가 cross-attention으로 주입된다.
- 시드 고정(`--seed`) 시 재현 가능하며, `--dtype`로 fp16/bf16 추론을 선택할 수 있다.

## 9. 기술적 선택 이유
- **x₀ = z_orig**: 조건부 이미지 생성 문제에서 목표 이미지를 직접 latent로 학습하면 추론 시 어떤 hairline mask가 들어와도 원본 스타일을 모사할 수 있어 일반화에 유리하다.
- **Mask의 이중 활용**: 공간적 정보(UNet 입력)와 전역 토큰(cond) 모두로 사용해 세부 hairline을 보존하면서 cross-attention이 전역 스타일과 모양을 동시에 조정할 수 있도록 했다.
- **Bald latent token**: 조건 이미지의 전체 스타일을 cond 토큰으로 제공해 피부 톤, 조명, 얼굴 구조의 일관성을 유지한다.
- **Accelerate + mixed precision**: 대규모 diffusion 모델 학습 비용을 줄이고, 분산/누적 학습 구성을 간결하게 관리한다.
- **DDIM 추론**: 빠른 샘플링과 guidance 조절이 가능해 사용자가 결과 품질과 속도 사이를 조절할 수 있다.


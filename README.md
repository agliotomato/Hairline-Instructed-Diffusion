# Hairline-Instructed Diffusion

> **대머리 이미지 + 의사 설계 헤어라인 마스크 → 자연스러운 머리 복원**  
> 의사가 설계한 머리선(hairline mask)에 맞춰 대머리 이미지를 자연스러운 머리 이미지로 복원하는 조건부 Stable Diffusion 모델입니다. 임의의 헤어라인 마스크와 텍스트 프롬프트를 통해 의학적·미용적 시뮬레이션에 활용할 수 있습니다.

<p align="center">
  <img src=assets/nlm.png/>
</p>

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [실험 진화 흐름](#실험-진화-흐름)
- [Track 1: SD 1.5 기반 학습](#track-1-sd-15-기반-학습)
  - [V4 아키텍처: Dual-Path Control](#v4-아키텍처-dual-path-control)
  - [V4_2: GAN Inversion & 비교 실험](#v4_2-gan-inversion--비교-실험)
- [Track 2: SD 3.5 기반 실험](#track-2-sd-35-기반-실험)
  - [SD 3.5 모델 분석 및 선정](#sd-35-모델-분석-및-선정)
  - [Masking Denoise: 하이브리드 파이프라인](#masking-denoise-하이브리드-파이프라인)
  - [Blur & 마스크 품질 개선](#blur--마스크-품질-개선)
  - [New Pipeline: Seamless 생성 전략](#new-pipeline-seamless-생성-전략)
  - [Debug: Color Preservation](#debug-color-preservation)
- [SD 1.5 vs SD 3.5 결과 비교](#sd-15-vs-sd-35-결과-비교)
- [상세 문서 링크](#상세-문서-링크)

---

## 프로젝트 개요

| 항목 | 내용 |
| :--- | :--- |
| **Task** | 대머리 이미지 + Semantic Mask → 헤어 생성 (Inpainting) |
| **Base Models** | Stable Diffusion v1.5 (U-Net CNN), SD 3.5 Medium (MMDiT Transformer) |
| **핵심 모듈** | TinyAdapter (Geometry), LatentIdentityNet (Identity), Latent Blending, SONIC |
| **Conditioning** | SegFace 기반 고정밀 Hair Semantic Mask |
| **데이터** | FFHQ 기반 원본/대머리/마스크 트리플 쌍 (~220장) |

---

## 실험 진화 흐름

```
[SD 1.5 Track]
V1 (단채널 마스크 입력)
  └─ V2 (z_orig 타겟 학습)
       └─ V3 (Innovated, 그러나 정보 붕괴 발생)
            └─ V4 ← Dual-Path: Tiny Encoder(Geometry) + LatentIdentityNet(Identity)
                 └─ V4_2 ← GAN Inversion Natural Filter + Nano Banana 비교

[SD 3.5 Track]
SD3.5 분석 (MMDiT, Rectified Flow)
  └─ Latent Blending 기반 Zero-shot Inpainting 검증
       └─ TinyAdapter (16ch → 128ch → 256ch) + SONIC 노이즈 최적화
            └─ SegFace 마스크 품질 개선 + Smart Blur (Lanczos + Peak Norm)
                 └─ Soft-Blending으로 Color Preservation 해결 중
```

---

## Track 1: SD 1.5 기반 학습

### V4 아키텍처: Dual-Path Control

기존 V3의 두 가지 핵심 문제를 해결하기 위해 설계된 이중 경로 구조입니다.

**해결한 문제:**
- **정보 붕괴 (Information Collapse)**: 512px 마스크를 64px로 단순 다운샘플링 시 고주파 디테일 손실
- **허위 상관관계 (Spurious Correlation)**: 모델이 헤어라인 마스크 대신 대머리 이미지의 피부 경계를 그대로 추론

<p align="center">
  <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/assets/diagram.png" width="70%"/>
</p>

#### 이중 경로 구조

| Stream | 입력 | 역할 | 특징 |
| :---: | :--- | :--- | :--- |
| **A: Geometry** | 512×512 Binary Mask | "어떤 모양의 머리카락?" | Tiny Encoder로 고해상도 마스크 압축 (512→64px, 1ch→320ch) |
| **B: Identity** | 64×64 Masked Bald Latent | "누구의 얼굴?" | LatentIdentityNet으로 신원 보존 (Stride=1, 압축 없음) |

```
Tiny Encoder 구조:
Input (1ch, 512²) → conv_in → blocks[0~5] → conv_out (320ch, 64²)
                                stride=2 다운샘플 3회 → 정보 손실 없이 공간 압축
```

#### 학습 설정

```bash
accelerate launch --mixed_precision="fp16" train_hairline_cond_v4.py \
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --output_dir "hairline_cond_v4_hybrid2" \
  --orig_dir "data/original_images" \
  --bald_dir "data/bald_images" \
  --mask_dir "data/semantic_masks" \
  --resolution 512 \
  --learning_rate 1e-5 \
  --train_batch_size 4 \
  --num_train_epochs 200
```

**학습 대상**: Tiny Encoder + LatentIdentityNet + Zero Convs만 학습 (SD U-Net / VAE / CLIP Frozen)

#### V4 결과

| Bald Input | Semantic Mask | Prompt | 결과 |
| :---: | :---: | :--- | :---: |
| <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/data/bald_images/test1.png" width="120"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/data/semantic_masks/test1.png" width="120"/> | "high quality, realistic hairstyle, detailed texture, 8k" | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_test_semantic/test1.png" width="200"/> |
| <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/data/bald_images/test1.png" width="120"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/data/semantic_masks/test1.png" width="120"/> | "low skin fade haircut, black hair, textured top, sharp hairline" | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_test_exp/test1_1.png" width="200"/> |
| <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/data/bald_images/test1.png" width="120"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/data/semantic_masks/test1.png" width="120"/> | "korean male two block haircut, dark brown hair, wavy textured fringe" | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_test_exp/test1_2.png" width="200"/> |

→ **상세 내용**: [V4 실험 문서](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/main/v4.md)

---

### V4_2: GAN Inversion & 비교 실험

#### 1. GAN Inversion Natural Filter

V4 결과물의 어색한 텍스처를 후처리로 보정하기 위해 StyleGAN의 Manifold Projection을 활용했습니다.

**원리**: VividHairStyler에서 Source/Structure/Appearance를 동일 이미지로 설정 → Identity Mapping 유도 → FFHQ 분포로 재구성 → Out-of-Distribution 아티팩트 제거

| V4 Output | V4 + GAN Inversion | Semantic Mask |
| :---: | :---: | :---: |
| <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_test_semantic/test1.png" width="160"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_gan_inversion/test1.png" width="160"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/data/semantic_masks/test1.png" width="160"/> |
| <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_test_semantic/test4.png" width="160"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_gan_inversion/test4.png" width="160"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/data/semantic_masks/test4.jpg" width="160"/> |

#### 2. Nano Banana 비교 (외부 모델 vs V4)

동일 프롬프트/마스크 조건에서 상용 헤어 생성 모델(Nano Banana)과 헤어라인 준수 능력을 비교했습니다.

| | V4 Output | V4 + GAN Inv | Nano Banana |
| :---: | :---: | :---: | :---: |
| **test1** | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_test_semantic/test1.png" width="140"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_gan_inversion/test1.png" width="140"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/nano_banana/test1_nb.png" width="140"/> |
| **test3** | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_test_semantic/test3.png" width="140"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_gan_inversion/test3.png" width="140"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/nano_banana/test3_nb.png" width="140"/> |

→ **상세 내용**: [V4_2 실험 문서](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/main/v4_2.md)

---

## Track 2: SD 3.5 기반 실험

### SD 3.5 모델 분석 및 선정

V1.5의 한계(색상 제어, 텍스트 이해력, 해상도)를 극복하기 위해 SD 3.5로 마이그레이션을 분석했습니다.

#### 모델 선정 근거

| 모델 | 파라미터 | 아키텍처 | 선정 |
| :--- | :---: | :--- | :---: |
| SD v1.5 | 0.98B | U-Net (CNN) | 현재 V4 베이스 |
| **SD 3.5 Medium** | **2.5B** | **MMDiT (Transformer)** | **✅ 선정** |
| SD 3.5 Large | 8.1B | MMDiT | VRAM 과다 |

**선정 이유**: Color Attribution 능력 6배 향상 (0.06 → 0.36), Position 정확도 대폭 상승, 소비자 GPU 구동 가능

#### MMDiT의 핵심 장점 (현 프로젝트 적용 관점)

- **Joint Attention**: 텍스트와 이미지 패치가 함께 Attention → 색상을 정확한 위치에 입히는 Color Attribution 향상
- **Rectified Flow**: 직선 경로 학습 → 적은 Step으로 고품질 생성 가능
- **Color Bleeding 감소**: SD 1.5에서 머리 색이 배경색을 따라가던 문제 해소 기대

<p align="center">
  <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/images/mmdit_overview.png" width="65%"/>
</p>

→ **상세 내용**: [SD 3.5 분석 문서](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/main/SD3.5.md)

---

### Masking Denoise: 하이브리드 파이프라인

SD 3.5 기반 **Zero-shot Latent Blending Inpainting** 검증 후, 세 모듈을 통합한 하이브리드 파이프라인을 구성했습니다.

#### 파이프라인 구성

```
[대머리 이미지] ──VAE Encode──► z_orig
[헤어 마스크]  ──TinyAdapter──► F_mask (기하학적 가이드)
                                    │
               SONIC 노이즈 최적화   │
               z_T* (배경 매칭 노이즈)│
                                    ▼
              ┌─────────────────────────────────┐
              │   SD 3.5 Denoising Loop (N steps)│
              │  x_input = x_noise + F_adapter   │ ← TinyAdapter 주입
              │  z_next = M·z_pred + (1-M)·z_bg  │ ← Latent Blending
              └─────────────────────────────────┘
                                    │
                                VAE Decode
                                    ▼
                             [최종 생성 이미지]
```

| 모듈 | 기술 | 역할 |
| :--- | :--- | :--- |
| **Geometry** | TinyAdapter (16/128/256ch) | 마스크 형상을 MM-DiT 입력에 Additive 주입 |
| **Texture Init** | SONIC (Spectral Optimization) | 배경과 주파수 특성이 맞는 초기 노이즈 생성 |
| **Identity** | Latent Blending | 매 Step마다 마스크 외부를 원본 Latent로 교체 |

#### TinyAdapter 채널별 비교

| Input | Mask | 16ch (V1) | 128ch (V2) | 256ch (V3) |
| :---: | :---: | :---: | :---: | :---: |
| <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/data/bald_images/01047.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/data/segmantic_masks/01047.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/01047_result.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/01047_result_v2.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/01047_v3.png" width="110"/> |
| <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/data/bald_images/01056.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/data/segmantic_masks/01056.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/01056_result.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/01056_result_v2.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/01056_v3.png" width="110"/> |

→ **상세 내용**: [Masking Denoise 실험 문서](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/denoise_exp/masking_denoise_exp.md)

---

### Blur & 마스크 품질 개선

#### 1. SegFace 도입 (마스크 정밀도 향상)

기존 BiSeNet 대비 경계선이 훨씬 선명한 SegFace를 도입하여 헤어라인 제어 정밀도를 높였습니다.

| 원본 | BiSeNet 마스크 | SegFace 마스크 |
| :---: | :---: | :---: |
| <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/data/original_images/01047.png" width="160"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/data/segmantic_masks/01047_c.png" width="160"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/data/segmantic_masks/01047.png" width="160"/> |
| <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/data/original_images/04731.png" width="160"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/data/segmantic_masks/04731_c.png" width="160"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/data/segmantic_masks/04731.png" width="160"/> |

#### 2. Smart Blur 전략 (계단 현상 완화)

이마라인(Edge Zone)과 볼륨 영역(Core Zone)을 분리하여 서로 다른 강도의 블러를 적용합니다.

```
Core Zone (볼륨):  Blur radius = 5.0  → 부드러운 연결감
Edge Zone (이마):  Blur radius ≤ 0.8  → 잔머리 디테일 보존
Peak Normalization: max(M_blurred) → 1.0  → 신호 강도 유지
```

#### 3. 누적 개선 비교 (SD 1.5 → 초기 SD 3.5 → 현재)

| 구분 | Semantic Mask | SD 1.5 (V4) | 초기 SD 3.5 | 현재 (Smart Blur) |
| :---: | :---: | :---: | :---: | :---: |
| **test1** | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/test_data/segmantic_masks/test1.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_test_semantic/test1.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/test1.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/native2/test1_smart.png" width="110"/> |
| **test4** | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/test_data/segmantic_masks/test4.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_test_semantic/test4.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/test4.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/native2/test4.png" width="110"/> |
| **test6** | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/test_data/segmantic_masks/test6.png" width="110"/> | — | — | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/native2/test6_smart.png" width="110"/> |

→ **상세 내용**: [Blur & 개선 문서](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/denoise_exp/blur.md)

---

### New Pipeline: Seamless 생성 전략

#### 1. Lanczos Resampling (계단 현상 근본 해결)

Bilinear 보간의 고주파 성분 손실 문제를 Lanczos Resampling으로 해결했습니다.

$$L(x) = \text{sinc}(x)\cdot\text{sinc}(x/a), \quad -a < x < a$$

- **Bilinear**: 1024px → 128px 다운샘플 시 머리카락 Edge 뭉개짐, Staircase 발생
- **Lanczos**: Sinc 함수 기반으로 고주파 성분 보존 → 선명한 경계 유지

#### 2. 이중 Zone 블러링 최적 파라미터

| 파라미터 | 값 | 역할 |
| :--- | :---: | :--- |
| Blur (Edge) | **0.8** | 이마라인 디테일 보존 |
| Blur Radius (Core) | **5.0** | Smart Blur 기본 반경 |
| Adapter Scale | **3.0** | Geometry 유도 강도 |
| Peak Normalization | **÷ max** | 블러 후 신호 강도 복원 |

#### 3. 개선 전후 마스크 비교

| Before | After |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/debug_smart_blur_final.png" width="280"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/debug_v2_final.png" width="280"/> |

#### 4. 생성 결과 비교

| Before (native2) | After (new pipeline) |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/native2/test4_smart.png" width="280"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/0125/test4.png" width="280"/> |

→ **상세 내용**: [New Pipeline 문서](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/denoise_exp/newpipeline.md)

---

### Debug: Color Preservation

#### 문제: 색상이 진해지는 현상

생성 후반부(Step 30+)에서 이미지가 어두워지거나 색이 포화되는 현상이 발생했습니다.

**원인 분석**:
1. 중간 타임스텝의 노이즈 섞인 Latent를 VAE 디코딩 시 Dynamic Range 불안정
2. `guidance_scale` 과다로 인한 색상 포화

SONIC 생성 과정 중간 단계 시각화:

| Step 000 | Step 010 | Step 020 | Step 030 | Step 049 |
| :---: | :---: | :---: | :---: | :---: |
| <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/0127/intermediate_steps/step_000.png" width="100"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/0127/intermediate_steps/step_010.png" width="100"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/0127/intermediate_steps/step_020.png" width="100"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/0127/intermediate_steps/step_030.png" width="100"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/0127/intermediate_steps/step_049.png" width="100"/> |

#### 해결: Soft-Blending

이진 마스크 대신 경계 영역에 연속값(0~1)을 갖는 Soft Mask를 적용하고, 생성 과정 자체에서 경계를 함께 만들어냅니다.

| ID | 원본 | 결과 (01/25, 이전) | 결과 (01/28, Soft-Blending) |
| :---: | :---: | :---: | :---: |
| **test1** | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/test_data/original_images/test1.png" width="130"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/0125/test1.png" width="130"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/0128/test1.png" width="130"/> |
| **test3** | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/test_data/original_images/test3.png" width="130"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/0125/test3.png" width="130"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/0128/test3.png" width="130"/> |

**현재 상태**: 얼굴 색 진해지는 문제 해결. 헤어 자연스러움과 Soft-Blending 적용 간의 Trade-off 해결 진행 중.

→ **상세 내용**: [Debug & Color Preservation 문서](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/denoise_exp/debug.md)

---

## SD 1.5 vs SD 3.5 결과 비교

| ID | Bald Input | Mask | SD 1.5 (V4, 320ch) | SD 3.5 (128ch) | SD 3.5 (256ch) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **test1** | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/test_data/bald_images/test1.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/test_data/segmantic_masks/test1.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_test_semantic/test1.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/test1.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/test1_v3.png" width="110"/> |
| **test3** | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/test_data/bald_images/test3.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/test_data/segmantic_masks/test3.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_test_semantic/test3.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/test3.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/test3_v3.png" width="110"/> |
| **test5** | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/test_data/bald_images/test5.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/test_data/segmantic_masks/test5.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/main/results/v4_test_semantic/test5.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/test5.png" width="110"/> | <img src="https://raw.githubusercontent.com/agliotomato/Hairline-Instructed-Diffusion/denoise_exp/results/final_hybrid/test5_v3.png" width="110"/> |

---

## 상세 문서 링크

### SD 1.5 Track

| 문서 | 내용 요약 |
| :--- | :--- |
| [v4.md](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/main/v4.md) | V4 Dual-Path 아키텍처 상세 설계, Tiny Encoder 스펙, 학습/추론 명령어, 결과 |
| [v4_2.md](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/main/v4_2.md) | GAN Inversion Natural Filter, 머리 색상 제어 메커니즘 분석, Nano Banana 비교 |

### SD 3.5 Track

| 문서 | 내용 요약 |
| :--- | :--- |
| [SD3.5.md](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/main/SD3.5.md) | MMDiT / Rectified Flow 논문 분석, 모델 선정 근거, Color Attribution 비교 |
| [masking_denoise_exp.md](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/denoise_exp/masking_denoise_exp.md) | Latent Blending 검증, TinyAdapter + SONIC 하이브리드 파이프라인 설계, Flow Matching Loss |
| [blur.md](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/denoise_exp/blur.md) | SegFace vs BiSeNet 마스크 품질 비교, Smart Blur 전략, SD1.5→SD3.5 누적 개선 비교 |
| [newpipeline.md](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/denoise_exp/newpipeline.md) | Lanczos Resampling, 이중 Zone 블러링, 최적 파라미터, Seamless 생성 방법론 |
| [debug.md](https://github.com/agliotomato/Hairline-Instructed-Diffusion/blob/denoise_exp/debug.md) | 색상 포화 현상 분석, SONIC 중간 단계 시각화, Soft-Blending 기법 |

# Stable-Hair V3: Latent IdentityNet + 헤어라인 마스크

이 문서는 **Latent IdentityNet** (ControlNet)을 통합하여 얼굴 정체성(Identity)과 배경을 완벽하게 보존하면서, **헤어라인 마스크**를 사용하여 헤어 생성을 유도함으로써 기존의 "정체성 손실" 문제를 해결하는 **V3 아키텍처**의 기술적 구현 내용을 설명합니다.

## 1. 아키텍처 개요 (Architecture Overview)

V3 아키텍처는 생성 가이드를 두 가지 경로로 분리하여 문제를 해결합니다:

1.  **구조 및 영역 가이드 (Main UNet)**:
    *   **입력**: 5채널 (4채널 Latent + 1채널 마스크).
    *   **역할**: 제공된 헤어라인 마스크를 기반으로 **어디에** 머리카락을 생성할지 결정합니다.
    *   **가중치**: Stable Diffusion v1.5에서 파인튜닝(Fine-tuned).

2.  **정체성 및 배경 보존 (Latent IdentityNet)**:
    *   **입력**: 4채널 Latent (Noisy Latent) + **대머리 프록시 Latent** (`z_bald`).
    *   **역할**: 입력 이미지(대머리 이미지)에서 고해상도 정체성 및 배경 특징을 추출하여 Main UNet에 주입합니다.
    *   **가중치**: **Pretrained Stable-Hair Stage 2** (또는 UNet Encoder)에서 초기화 후 동결(Frozen) 또는 파인튜닝.

---

## 2. 학습의 목적: 무엇을 배우는가? 

이번 V3 학습의 핵심은 **"Main UNet이 IdentityNet의 정보를 받아들이는 법"**을 배우는 것입니다.

| 모델 컴포넌트 | 상태 (Status) | 역할 및 학습 내용 |
| :--- | :--- | :--- |
| **Latent IdentityNet** | **Frozen / Pretrained** | 이미 학습된 모델(Stage 2)을 가져와서 사용합니다. 대머리 이미지에서 얼굴/배경 특징을 추출하여 Main UNet에 전달합니다. (이번 학습에서 가중치가 크게 변하지 않거나 고정됨) |
| **Main UNet** | **Trainable (학습 대상)** | 기존에는 마스크만 보고 머리를 그렸지만, 이제는 **IdentityNet이 주는 힌트(얼굴/배경 정보)를 섞어서** 머리카락을 그리는 법을 새로 배웁니다. |

### 학습 원리 (Training Mechanism)
학습 과정에서 모델은 **원본 이미지(Original Latent)**를 정답지(Ground Truth)로 삼아 복원하는 것을 목표로 합니다.
1.  **문제**: 원본에 노이즈가 섞인 상태(`Noisy Latent`)를 입력받습니다.
2.  **힌트**: **IdentityNet이 제공하는 얼굴 정보**(`z_bald`)와 **마스크가 지정하는 영역 정보**를 단서로 받습니다.
3.  **목표**: 이 단서들을 조합하여 노이즈를 제거하고, **원래의 머리카락이 있는 상태(Original Latent)로 되돌리는(Denoising)** 방법을 학습합니다.

즉, **IdentityNet은 도구로 가져다 쓰고**, **Main UNet을 파인튜닝**하여 두 정보(마스크 + 정체성)를 조화롭게 사용하는 능력을 기르는 과정입니다.

---

## 3. 학습 스크립트: `train_hairline_cond_v3.py`

이 스크립트는 Latent IdentityNet이 제공하는 특징을 활용하면서 마스크 영역 내에 머리카락을 생성하도록 Main UNet을 학습시킵니다.

### 주요 기능
*   **이중 입력 스트림 (Dual Input Streams)**:
    *   **Main UNet**: `torch.cat([noisy_latents, mask_latents], dim=1)`을 입력으로 받습니다.
    *   **IdentityNet**: `noisy_latents`와 `controlnet_cond` (`z_bald`)를 입력으로 받습니다.
*   **특징 주입 (Feature Injection)**:
    *   IdentityNet의 Down-block 및 Mid-block 출력이 `down_block_additional_residuals` 및 `mid_block_additional_residual`을 통해 Main UNet의 해당 블록에 더해집니다.
*   **사전 학습된 가중치 로드 (Pretrained Weights Loading)**:
    *   **Stable-Hair Stage 2** 체크포인트 (`pytorch_model_2.bin`)를 로드하여 강력한 정체성 보존 기능을 즉시 활용할 수 있습니다.
    *   *수정 사항*: 가중치 호환성(4채널 입력)을 보장하기 위해 Main UNet의 입력 채널을 수정하기 *전에* ControlNet을 초기화하도록 로직을 수정했습니다.
        ```python
        # 1. Initialize Latent IdentityNet (Takes 4-channel input)
        if args.controlnet_model_name_or_path:
            controlnet = ControlNetModel.from_pretrained(...)
        else:
            controlnet = ControlNetModel.from_unet(unet, load_weights_from_unet=True)

        # 2. Enable 5-channel input for Main UNet (4 latent + 1 mask)
        # MUST be done AFTER ControlNet init to avoid channel mismatch error
        unet = enable_hairline_conditioning(unet, mask_channels=1)
        ```

### 학습 명령어 (Lambda AI)
```bash
python3 train_hairline_cond_v3.py \
  --orig_dir="data/original_images" \
  --bald_dir="data/bald_images" \
  --mask_dir="data/only_forehead_line" \
  --controlnet_model_name_or_path="checkpoints/stage2/pytorch_model_2.bin" \
  --output_dir="hairline_cond_v3_run1" \
  --train_batch_size=2 \
  --num_train_epochs=50 \
  --learning_rate=1e-5 \
  --checkpoints_total_limit=2 \
  --report_to="none" \
  --dataloader_num_workers=0
```
*(참고: `--dataloader_num_workers=0` 옵션은 Docker 환경에서 공유 메모리(Shared Memory) 부족 에러를 방지하기 위해 사용됩니다.)*

---

## 4. 추론 스크립트: `infer_hairline_cond_v3.py`

추론 스크립트는 마스크 가이드와 정체성 보존을 결합하여 생성 프로세스를 수행합니다.

### 워크플로우
1.  **전처리 (Preprocessing)**:
    *   **대머리 이미지**: VAE Latent 공간으로 인코딩 $\rightarrow$ `z_bald`.
    *   **마스크**: $64 \times 64$ Latent 크기로 리사이즈 $\rightarrow$ `mask_latent`.
2.  **Latent 초기화 (Latent Initialization)**:
    *   **랜덤 노이즈 초기화** (`--init_latent noise`): 표준적인 Diffusion 방식대로 순수한 가우시안 노이즈에서 시작합니다.
3.  **샘플링 루프 (Sampling Loop)**:
    *   각 타임스텝 $t$에서:
        1.  **IdentityNet Forward**: `latents`와 `z_bald`를 전달하여 정체성 특징을 추출합니다.
        2.  **Main UNet Forward**: `latents`와 `mask_latent`를 연결(Concat)하여 5채널 입력을 만들고, IdentityNet에서 추출한 특징을 주입받습니다.
        3.  **노이즈 예측**: Main UNet은 마스크(구조)와 IdentityNet(내용)의 가이드를 받아 노이즈를 예측합니다.

### 추론 명령어 예시
```bash
python3 infer_hairline_cond_v3.py \
  --bald_path "data/bald_images/01047.png" \
  --mask_path "data/only_forehead_line/01056.png" \
  --controlnet_path "hairline_cond_v3_run1/controlnet" \
  --unet_path "hairline_cond_v3_run1/unet" \
  --out_dir "result/v3_test" \
  --init_latent noise \
  --noise_strength 1.0
```


<div align="center">
  <img src="data/original_images/01047.png" width="23%" />
  <img src="data/bald_images/01047.png" width="23%" />
  <img src="data/only_forehead_line/01047.png" width="23%" />
  <img src="result/v3_test/sample_20251125_071202_000.png" width="23%" />
</div>

<div align="center">
  <img src="data/original_images/01047.png" width="23%" />
  <img src="data/bald_images/01047.png" width="23%" />
  <img src="data/only_forehead_line/01056.png" width="23%" />
  <img src="result/v3_test/sample_20251125_071706_000.png" width="23%" />
</div>

<div align="center">
  <img src="data/original_images/01047.png" width="23%" />
  <img src="data/bald_images/01047.png" width="23%" />
  <img src="data/only_forehead_line/01056.png" width="23%" />
  <img src="result/v3_test/sample_20251125_072428_000.png" width="23%" />
</div>

<div align="center">
  <img src="data/original_images/01056.png" width="23%" />
  <img src="data/bald_images/01056.png" width="23%" />
  <img src="data/only_forehead_line/01047.png" width="23%" />
  <img src="result/v3_test/sample_20251125_073051_000.png" width="23%" />
</div>

<div align="center">
  <img src="data/original_images/01056.png" width="23%" />
  <img src="data/original_images/01056.png" width="23%" />
  <img src="data/only_forehead_line/01047.png" width="23%" />
  <img src="result/v3_test/sample_20251125_073531_000.png" width="23%" />
</div>

# SD3.5 Hybrid Hair Generation Implementation

본 문서는 **Hairline-Instructed Diffusion** (SD 3.5 기반)의 구현 과정, 전략, 그리고 실행 방법을 정리한 문서입니다.

## 1. Architecture: Hybrid Dual-Stream ControlNet

우리는 **Geometry(형태)**와 **Identity(신원/맥락)**를 동시에 제어하기 위해 두 개의 ControlNet을 사용하는 **Hybrid Architecture**를 구현했습니다.

### Base Model: MM-DiT (Multimodal Diffusion Transformer)
*   **Backbone:** `SD3Transformer2DModel` (Stable Diffusion 3.5 Medium)
*   **Architecture:** **MM-DiT** 구조를 사용하여 이미지 Latent와 텍스트 임베딩(T5+CLIP)을 효과적으로 융합합니다.
*   **State:** 학습 중에는 Frozen 상태이며, 두 개의 ControlNet이 이 MM-DiT 구조를 복제/참조하여 학습됩니다.

### Stream A: Geometry Control (Spatial)
*   **역할:** 머리카락의 **정확한 형태(Shape)**와 **흐름(Flow)**을 제어합니다.
*   **입력:** 1-Channel Mask (Background=0, Face=0.5, Hair=1.0)
*   **구현:** 
    *   `SD3ControlNetModel` (Transformer Backbone 복사 초기화)
    *   `extra_conditioning_channels=1`
    *   입력 마스크는 VAE를 거치지 않고 리사이징 후 Latent Space에 *Concatenate* 됩니다.

### Stream B: Identity Control (Latent)
*   **역할:** 얼굴의 **톤(Tone)**, **생김새(Identity)**, **조명(Lighting)**을 유지하여 자연스러운 합성을 유도합니다.
*   **입력:** 16-Channel VAE Latents (Masked Bald Image)
*   **구현:**
    *   `SD3ControlNetModel` (Manual Initialization)
    *   `extra_conditioning_channels=16`
    *   대머리 이미지는 VAE를 통과하여 16채널 Latent로 변환된 후 입력됩니다.

### Component Training Status

| Component | Status | Details |
| :--- | :---: | :--- |
| **VAE** | ❄️ Frozen | Latent Encoding을 위해 사전 학습된 SD3.5 VAE 사용 |
| **Text Encoder 1** (CLIP-L) | ❄️ Frozen | 사전 학습됨 |
| **Text Encoder 2** (OpenCLIP-G) | ❄️ Frozen | 사전 학습됨 |
| **Text Encoder 3** (T5-XXL) | ❄️ Frozen | 사전 학습됨 (FP16/BF16) |
| **Transformer Backbone** (MM-DiT) | ❄️ Frozen | Stable Diffusion 3.5 Medium 기본 모델 |
| **ControlNet Stream A** (Geometry) | 🔥 **Trainable** | MM-DiT에서 초기화됨, 형태(Shape)와 흐름(Flow) 학습 |
| **ControlNet Stream B** (Identity) | 🔥 **Trainable** | MM-DiT에서 초기화됨, 신원(Identity)과 질감(Texture) 학습 |

---

## 2. Training Strategy: "Smart Mask" & Masked Loss

부자연스러운 "스티커 같은 합성"을 방지하고, 이마 라인(Hairline)의 디테일을 살리기 위해 **Smart Mask Augmentation**과 **Masked Loss** 전략을 도입했습니다.

### "Smart Mask" Logic (`hairline_dataset_v2.py`)
학습 데이터 로딩 시 50% 확률로 다음 로직이 적용됩니다:

1.  **Hair Blur (Volume):** 머리카락 전체 영역에 Gaussian Blur를 적용하여 가장자리가 부드럽게 배경과 섞이도록 유도합니다.
2.  **Face Protection (Hairline):** 얼굴(Face) 영역을 중심으로 "Protection Zone"을 설정합니다.
3.  **Localized Composition:**
    *   **헤어라인 근처:** 선명한(Sharp) 마스크 유지 (정확한 이식)
    *   **그 외 외곽:** 흐릿한(Blur) 마스크 적용 (자연스러운 블렌딩)

### Masked Loss (Localized Optimization)
*   **Noise:** 노이즈는 **이미지 전체($full\ image$)**에 추가됩니다. (Global Noise)
    *   이유: 배경과 얼굴의 문맥(Context)을 모델이 이해해야 자연스러운 연결이 가능하기 때문입니다.
*   **Loss:** 오차 계산은 **머리카락 영역($mask$)**에서만 수행합니다. (Local Loss)
    *   이유: 모델이 배경이나 얼굴을 복원하는 데 능력을 낭비하지 않고, **헤어 생성**에만 집중하도록 강제합니다. Identity ControlNet이 배경 유지를 담당한다고 가정합니다.


### Training Target (y)
본 학습에서 모델이 예측해야 하는 정답(Target, $y$)은 **머리카락이 포함된 원본 이미지(Original Image)**입니다.
*   **Goal:** 노이즈 상태($z$)에서 시작하여 조건(Identity, Mask, Prompt)을 만족하는 **원본 이미지($y$)**를 복원하는 것.
*   **Flow Matching:** 수식적으로는 노이즈에서 원본으로 향하는 **최단 경로의 속도 벡터(Velocity)**를 학습합니다.
*   **Loss:** 예측된 이미지(Velocity)와 원본 이미지(Ground Truth) 간의 차이를 줄이는 방향으로 학습됩니다.

### Training Optimizations (High VRAM)
H100/GH200 (80GB+) 환경에 맞춰 최적화되었습니다.
*   **Full GPU Loading:** Text Encoder (T5-XXL), Transformer를 모두 GPU에 상주시켜 CPU 병목 제거.
*   **Precision:** `bfloat16` (BF16) 사용으로 연산 속도 및 안정성 확보.
*   **Optimizer:** `bitsandbytes.optim.AdamW8bit` 사용으로 VRAM 절약.

---

## 3. Training Command (H100 / GH200 Optimized)

아래 명령어는 **80GB 이상 VRAM**을 가진 A100/H100/GH200 인스턴스에서 **1024px** 해상도로 학습하기 위한 최적 설정입니다.

```bash
# 가상환경 활성화
source venv/bin/activate

# 필수: H100에서는 bf16이 필수적입니다.
python train_hairline_cond_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium" \
  --orig_dir="data/original_images" \
  --bald_dir="data/bald_images" \
  --mask_dir="data/semantic_masks" \
  --output_dir="output/hairline_sd3_run1" \
  --resolution=1024 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --mixed_precision="bf16" \
  --num_train_epochs=50 \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=3
```

---

## 4. Inference Strategy

추론 시에도 학습과 동일한 "Smart Mask" 전처리를 적용하여 최상의 결과를 얻습니다.

### Key Parameters
*   **Geometry Scale:** `0.5 ~ 0.8` (낮을수록 창의적, 높을수록 마스크 준수)
*   **Identity Scale:** `0.8 ~ 1.0` (강하게 주어 얼굴이 변하지 않도록 함)
*   **Guidance Scale (CFG):** `4.0 ~ 7.0`

### Workflow
1.  **Input:** 대머리 이미지 + 원하는 헤어스타일 텍스트 + (옵션) 가이드 마스크
2.  **Process:**
    *   대머리 이미지는 VAE 인코딩되어 Stream B로 전달.
    *   가이드 마스크(혹은 자동 생성된 마스크)는 Stream A로 전달.
3.  **Generation:** SD3.5가 두 ControlNet의 가이드를 받아 고해상도 헤어 생성.

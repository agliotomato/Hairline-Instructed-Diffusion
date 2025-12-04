# Semantic Mask Integration Pipeline

이 문서는 **1채널 시맨틱 마스크(Semantic Mask)**를 사용하여 ControlNet을 학습하고 추론하는 전체 파이프라인을 상세하게 설명합니다.

## 1. 데이터 준비 (Data Preparation)

### 마스크 형식 (Mask Format)
*   **채널 수**: 1채널 (Grayscale)
*   **값의 의미 (Pixel Values)**:
    *   `0`: 배경 (Background)
    *   `128`: 얼굴 피부 (Face Skin)
    *   `255`: 헤어 (Hair)
*   **주의사항**: 안티앨리어싱(Anti-aliasing)이 적용되지 않은, 경계가 뚜렷한 마스크여야 합니다.

---

## 2. 데이터 로딩 (Dataset Loading)

### 파일: `utils/hairline_dataset_v2.py`

데이터셋 클래스 `HairlineDatasetV2`는 이미지와 마스크를 불러와 전처리를 수행합니다.

#### 핵심 변경 사항: `NEAREST` 보간법
시맨틱 마스크는 연속적인 값이 아닌 **이산적인 클래스(0, 128, 255)**를 가집니다. 일반적인 이미지 리사이징(`BILINEAR`, `BICUBIC`)을 사용하면 `64`, `192`와 같은 중간값이 생성되어 모델에게 혼란을 줍니다. 이를 방지하기 위해 **`NEAREST` (최근접 이웃)** 보간법을 사용합니다.

```python
self.mask_transform = transforms.Compose(
    [
        # 중요: NEAREST를 사용하여 0, 128, 255 값을 그대로 유지
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ]
)
```

---

## 3. 학습 파이프라인 (Training Pipeline)

### 파일: `train_hairline_cond_v3.py`

학습 시 ControlNet은 **5채널 입력**을 받습니다.

#### 입력 구조 (Input Structure)
1.  **Bald Proxy Latent (4채널)**: VAE로 인코딩된 대머리 이미지의 Latent 표현.
2.  **Semantic Mask (1채널)**: 64x64 크기로 리사이징된 시맨틱 마스크.

#### Latent 리사이징
학습 과정에서 512x512 마스크를 64x64 Latent 크기로 줄일 때도 `NEAREST`를 사용해야 합니다.

```python
# 마스크를 Latent 크기(64x64)로 줄일 때도 값 보존을 위해 nearest 사용
mask_latents = F.interpolate(
    hair_masks, size=noisy_latents.shape[-2:], mode="nearest"
)
# 5채널로 결합: [Bald Latent(4) + Mask(1)]
controlnet_cond = torch.cat([z_bald, mask_latents], dim=1)
```

#### Weight Surgery (가중치 수술)
기존 4채널 입력을 받던 ControlNet의 첫 번째 레이어(`conv_in_2`)를 5채널로 확장합니다.
*   **기존 4채널**: 사전 학습된 가중치 유지.
*   **추가 1채널 (마스크)**: `0`으로 초기화하여 학습 초기에는 기존 모델의 동작을 방해하지 않도록 함.

---

## 4. 추론 파이프라인 (Inference Pipeline)

### 파일: `infer_hairline_cond_v3.py`

학습된 모델을 사용하여 이미지를 생성할 때도 동일한 전처리 과정을 거칩니다.

#### 전처리 (Preprocessing)
```python
def preprocess_mask(path: str, resolution: int) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    # 추론 시에도 NEAREST 사용
    mask = mask.resize((resolution, resolution), Image.NEAREST)
    tensor = transforms.ToTensor()(mask).unsqueeze(0)
    return torch.clamp(tensor, 0.0, 1.0)
```

#### 파이프라인 흐름
1.  **입력**: 대머리 이미지, 시맨틱 마스크.
2.  **인코딩**: 대머리 이미지를 VAE로 인코딩 -> `z_bald` (Latent).
3.  **마스크 준비**: 시맨틱 마스크를 64x64로 리사이징 (`mode="nearest"`).
4.  **결합**: `z_bald`와 `mask_latent`를 결합하여 5채널 조건 생성.
5.  **생성**: ControlNet이 5채널 조건을 받아 UNet의 생성을 가이드함.

---

## 요약 (Summary)

| 단계 | 주요 작업 | 핵심 포인트 |
| :--- | :--- | :--- |
| **데이터셋** | 마스크 로드 및 리사이징 | `InterpolationMode.NEAREST` 사용 |
| **학습** | Latent 크기로 변환 및 입력 결합 | `mode="nearest"`, 5채널 입력 (4+1) |
| **추론** | 마스크 전처리 및 주입 | 학습과 동일한 `NEAREST` 전처리 유지 |

이 파이프라인을 통해 모델은 배경, 얼굴, 헤어 영역을 명확하게 구분하여 학습하고 생성할 수 있습니다.

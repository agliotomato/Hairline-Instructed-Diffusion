# Stable-Hair V3 Innovation: Latent IdentityNet Weight Surgery

## 1. 핵심 전략: Weight Surgery

기존의 Latent IdentityNet은 4채널 입력(대머리 이미지)만 받을 수 있었습니다. 우리는 이를 **5채널 입력(대머리 + 마스크)**으로 확장하면서도, 기존의 학습된 능력을 100% 보존하는 전략을 사용했습니다.

### 구현 원리
1.  **Main UNet 보존**: Main UNet은 **순정 4채널 입력**을 그대로 유지합니다. (기존 SD 1.5 호환성 유지)
2.  **IdentityNet 확장**: IdentityNet의 입력 레이어(`conv_in_2`)를 4채널에서 5채널로 확장합니다.
3.  **Zero Initialization**:
    *   **채널 0~3 (대머리)**: 기존 Pretrained 가중치를 그대로 복사합니다. (기존 지식 보존)
    *   **채널 4 (마스크)**: **0으로 초기화**합니다. (초기 충격 방지 및 점진적 학습 유도)

```python
# Weight Surgery Code Snippet
# 주의: ControlNetModel의 입력 레이어 이름은 'conv_in_2'입니다.
old_conv_weight = controlnet.conv_in_2.weight.data # Shape: [320, 4, 3, 3]
new_conv_weight = torch.zeros((320, 5, 3, 3), device=controlnet.device)

# 1. 기존 지식 복사 (Preservation)
new_conv_weight[:, :4, :, :] = old_conv_weight

# 2. 새 채널 0으로 초기화 (Gradual Learning)
new_conv_weight[:, 4:, :, :] = 0.0

# 3. 이식
controlnet.conv_in_2.weight.data = new_conv_weight
```

---

## 2. 학습 (Training)

학습 시 모델은 다음과 같은 입력을 받습니다:
*   **Main UNet**: `Noisy Latent` (4채널)
*   **IdentityNet**: `Bald Proxy` (4채널) + `Hairline Mask` (1채널) = **5채널 Condition**

### 학습 명령어 (Lambda AI)
```bash
python3 train_hairline_cond_v3.py \
  --orig_dir="data/original_images" \
  --bald_dir="data/bald_images" \
  --mask_dir="data/only_forehead_line" \
  --controlnet_model_name_or_path="stage2/pytorch_model_2.bin" \
  --output_dir="hairline_cond_v3_innovation" \
  --train_batch_size=2 \
  --num_train_epochs=50 \
  --learning_rate=1e-5 \
  --checkpoints_total_limit=2 \
  --report_to="none" \
  --dataloader_num_workers=0
```

### 트러블슈팅 (Troubleshooting)
*   **AttributeError: 'ControlNetModel' object has no attribute 'controlnet_cond_embedding'**:
    *   **원인**: 커스텀 `ControlNetModel` 클래스는 일반적인 Diffusers 구조와 달리 `controlnet_cond_embedding`이 없고, 직접 `conv_in_2` 레이어를 사용합니다.
    *   **해결**: Weight Surgery 대상을 `controlnet.conv_in_2.weight`로 수정했습니다.
*   **SyntaxError: keyword argument repeated**:
    *   **원인**: `controlnet` 호출 시 `encoder_hidden_states` 인자가 중복되었습니다.
    *   **해결**: 중복된 인자를 제거했습니다.

---

## 3. 추론 (Inference)

추론 시에도 학습과 동일하게 IdentityNet에 5채널 입력을 제공해야 합니다.

### 추론 스크립트 (`infer_hairline_cond_v3.py`) 업데이트 완료
현재 스크립트는 Innovation 전략에 맞춰 다음과 같이 수정되었습니다:
1.  **Main UNet**: 순정 4채널 입력 사용 (`enable_hairline_conditioning` 제거).
2.  **IdentityNet 입력**: `Bald Proxy` + `Mask`를 결합하여 **5채널 Condition** 생성.
3.  **마스크 처리**: `nearest` 모드로 리사이징하여 경계선 보존.

### 추론 명령어 예시
```bash
python3 infer_hairline_cond_v3.py \
  --bald_path "data/bald_images/01047.png" \
  --mask_path "data/only_forehead_line/01047.png" \
  --controlnet_path "hairline_cond_v3_innovation/controlnet" \
  --unet_path "hairline_cond_v3_innovation/unet" \
  --out_dir "result/v3_innov_test" \
  --init_latent noise \
  --noise_strength 1.0

```

<div align="center">
  <img src="data/original_images/01047.png" width="23%" />
  <img src="data/bald_images/01047.png" width="23%" />
  <img src="data/only_forehead_line/01047.png" width="23%" />
  <img src="result/v3_innov_test/b01047_m01047/sample_20251127_011727_000.png" width="23%" />
</div>

<div align="center">
  <img src="data/original_images/01047.png" width="23%" />
  <img src="data/bald_images/01047.png" width="23%" />
  <img src="data/only_forehead_line/01056.png" width="23%" />
  <img src="result/v3_innov_test/b01047_m01056/sample_20251127_010438_000.png" width="23%" />
</div>

<div align="center">
  <img src="data/original_images/01047.png" width="23%" />
  <img src="data/bald_images/01047.png" width="23%" />
  <img src="data/only_forehead_line/01057.png" width="23%" />
  <img src="result/v3_innov_test/b01047_m01057/sample_20251127_010520_000.png" 
  width="23%" />
</div>

<div align="center">
  <img src="data/original_images/01056.png" width="23%" />
  <img src="data/bald_images/01056.png" width="23%" />
  <img src="data/only_forehead_line/01047.png" width="23%" />
  <img src="result/v3_innov_test/b01056_m01047/sample_20251127_010553_000.png" width="23%" />
</div>

<div align="center">
  <img src="data/original_images/01056.png" width="23%" />
  <img src="data/bald_images/01056.png" width="23%" />
  <img src="data/only_forehead_line/01057.png" width="23%" />
  <img src="result/v3_innov_test/b01056_m01057/sample_20251127_010829_000.png" width="23%" />
</div>

<div align="center">
  <img src="data/original_images/01057.png" width="23%" />
  <img src="data/bald_images/01057.png" width="23%" />
  <img src="data/only_forehead_line/01047.png" width="23%" />
  <img src="result/v3_innov_test/b01057_m01047/sample_20251127_010908_000.png" width="23%" />
</div>

<div align="center">
  <img src="data/original_images/01057.png" width="23%" />
  <img src="data/bald_images/01057.png" width="23%" />
  <img src="data/only_forehead_line/01056.png" width="23%" />
  <img src="result/v3_innov_test/b01057_m01056/sample_20251127_010944_000.png" width="23%" />
</div>

<div align="center">
  <img src="data/original_images/01057.png" width="23%" />
  <img src="data/bald_images/01057.png" width="23%" />
  <img src="data/only_forehead_line/01057.png" width="23%" />
  <img src="result/v3_innov_test/b01057_m01057/sample_20251127_011019_000.png" width="23%" />
</div>






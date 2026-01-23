# Masking Denoise

## 1. 실험 개요
Stable Diffusion 3.5 (SD3.5)의 기본 모델만을 사용하여 **Latent Blending** 기법을 통한 인페인팅(Inpainting) 가눙성을 검증합니다. 별도의 ControlNet이나 Inpainting 전용 모델 없이, 생성 과정에서 마스크 영역 밖을 원본으로 강제 교체하여 배경과 얼굴을 보존하는 것이 목표입니다.

## 2. 실행 명령어

```bash
python scripts/test_sd3_inpainting_basic.py \
  --image_path "data/bald_images/01047.png" \
  --mask_path "data/semantic_masks/01047.png" \
  --prompt "a photo of a man with short brown hair, high quality, realistic" \
  --output_path "output/denoise_exp/01047.png" \
  --steps 28 \
  --guidance_scale 5.0 \
  --strength 1.0
```

### 파라미터 설명
*   `--image_path`: 원본 대머리 이미지 (배경 및 얼굴 보존용 소스).
*   `--mask_path`: 헤어 영역 마스크 (흰색=생성, 검은색=보존).
*   `--strength`: `1.0`. 100% 인페인팅을 의미하며, 마스크 영역 내부는 완전히 새로 생성합니다.

## 3. Proposed Pipeline: SD 3.5 Hybrid Architecture

본 프로젝트가 제안하는 파이프라인은 **SD 3.5의 강력한 생성 능력**, **TinyAdapter의 정밀한 형상 제어**, **SONIC의 초기 노이즈 최적화**, 그리고 **Latent Blending의 배경 보존**을 결합한 하이브리드 아키텍처입니다.

### 3.1. Core Components (핵심 구성 요소)

파이프라인은 크게 세 가지 핵심 모듈로 구성됩니다.

1.  **Geometry Stream (TinyAdapter)**
    *   **역할**: 헤어 마스크($M$)를 입력받아 **"어디에 머리카락이 있어야 하는지"**에 대한 기하학적 가이드($F_{mask}$)를 생성합니다.
    *   **특징**: 경량화된 CNN 구조(Tiny Encoder)를 사용하여 SD 3.5의 VRAM 부담을 최소화하면서도, 블러링된 Soft Mask를 통해 자연스러운 헤어라인을 유도합니다.
    *   **통합 방식(Integration)**: MM-DiT 블록의 Joint Attention 입력단(Before Normalization)에 Feature를 Additive($x + F_{mask}$) 방식으로 주입합니다.

2.  **Texture Initialization (SONIC)**
    *   **역할**: 무작위 가우시안 노이즈 대신, 원본 이미지의 주파수 특성을 반영한 **최적화된 노이즈($z_T^*$)**를 생성의 시작점으로 사용합니다.
    *   **효과**: 생성된 헤어 텍스처가 원본 얼굴 피부톤 및 조명과 이질감 없이 조화를 이루도록 합니다.

3.  **Identity Preservation (Latent Blending)**
    *   **역할**: 생성 과정(Denoising Loop) 동안 마스크 외부 영역을 원본 이미지의 Latent로 강제 교체(Replacement)합니다.
    *   **효과**: 얼굴과 배경의 ID Identity가 100% 보존되며, VAE 인코딩/디코딩 과정에서의 손실을 최소화합니다.

### 3.2. Process Workflow (프로세스 흐름)

전체 생성 과정은 다음과 같은 순서로 진행됩니다.

1.  **Preprocessing**:
    *   **Latent Encoding**: 대머리 원본 이미지($I_{bald}$)를 VAE로 인코딩하여 $z_{orig}$ 생성.
    *   **Mask Preparation**: 헤어 마스크($M$)를 Latent 크기로 리사이징하고, 경계를 부드럽게 처리(Blurring).

2.  **Noise Optimization (SONIC Step)**:
    *   $z_{orig}$와 텍스트 프롬프트를 기반으로 짧은 최적화 단계를 수행하여, 배경과 자연스럽게 섞일 수 있는 초기 노이즈 $z_T^*$를 생성합니다.

3.  **Denoising with Dual-Control (Main Loop)**:
    *   **Step 1: Adapter Injection**: TinyAdapter가 마스크 특징($F_{mask}$)을 추출하여 MM-DiT의 입력 노이즈에 더해줍니다. (Geometry Control).
    *   **Step 2: Noise Prediction**: SD 3.5 Transformer가 노이즈를 예측($\epsilon_\theta$).
    *   **Step 3: Latent Blending**: 예측된 Latent에서 마스크 외부(배경/얼굴) 영역을 잘라내고, 원본 이미지에 노이즈를 섞은 $z_{bg}$로 교체합니다.
        $$ z_{next} = M \cdot z_{pred} + (1-M) \cdot z_{bg} $$

4.  **Final Output**:
    *   최종 Denoising이 완료된 Latent $z_0$를 VAE Decoding하여 고해상도 이미지를 얻습니다.

## 4. 개선된 실험: SONIC + Blending Integration
단순 랜덤 노이즈 대신, **SONIC (Spectral Optimization)** 기술을 적용하여 "배경과 어울리는 초기 노이즈"를 사용하면 훨씬 자연스러운 결과를 얻을 수 있습니다.

SONIC의 핵심 아이디어는 **"이미지의 주파수 성분을 노이즈로 변환하여 배경과 일치하는 노이즈를 생성"**하는 것입니다.

SONIC 설명 : 업데이트 예정(논문 읽는 중)

### 실행 명령어 (Command)

```bash
python scripts/test_sd3_sonic_blended.py \
  --image_path "data/bald_images/01047.png" \
  --mask_path "data/semantic_masks/01047.png" \
  --prompt "a photo of a man with short brown hair, high quality, realistic" \
  --output_path "output/sonic/01047_blended.png" \
  --opt_steps 15 \
  --lr 0.01 \
  --mask_blur 3.0
```

### 주요 파라미터
*   `--opt_steps 15`: 인페인팅 시작 전, 노이즈를 15번 최적화하여 배경과 매칭시킵니다.
*   `--mask_blur 3.0`: 마스크 경계를 부드럽게 처리하여(Soft Mask) 이질감을 줄입니다.


## 5. 실험 결과 (Results)

| ID | Bald Image (Source) | Semantic Mask | Initial | SONIC |
| :---: | :---: | :---: | :---: | :---: |
| **01047** | ![Bald](data/bald_images/01047.png) | ![Mask](data/segmantic_masks/01047.png) | ![Initial](results/denoise_exp/01047_fixed.png) | ![SONIC](results/sonic/01047_blended.png) |
| **01056** | ![Bald](data/bald_images/01056.png) | ![Mask](data/segmantic_masks/01056.png) | ![Initial](results/denoise_exp/01056_fixed.png) | ![SONIC](results/sonic/01056_blended.png) |
| **01057** | ![Bald](data/bald_images/01057.png) | ![Mask](data/segmantic_masks/01057.png) | ![Initial](results/denoise_exp/01057_fixed.png) | ![SONIC](results/sonic/01057_blended.png) |

## 6. 결론

원하는 곳에만 noise 주는 것이 **"가능하다!"**

실험 결과, 
*   **배경 보존**: 원본 대머리 이미지의 얼굴과 배경이 거의 완벽하게 유지
*   **자연스러운 연결**: 생성된 헤어가 이마 라인과 자연스럽게 어우러짐
*   **이마라인 보존**: 이 부분은 조건 주입을 통해 보완 가능 
    
### 6.1. 최종 아키텍처

| 구성 요소 | 기술 (Technology) | 역할 및 효과 |
| :--- | :--- | :--- |
| **Texture Init** | **Spectral Seed Optimization (SONIC)** | 배경과 일치하는 최적의 노이즈($z_T^*$)를 확보하여 이질감 원천 봉쇄 |
| **Identity** | **Latent Blending** | 최적화 중 'Drift' 현상을 막고 얼굴/배경을 물리적으로 100% 보존 |
| **Geometry** | **Tiny Adapter** | **16ch (V1)** vs **128ch (V2)** vs **256ch (V3)**: 마스크 모양을 Latent Space에 주입하여 헤어라인 형상 제어 |

## 7. Tiny Encoder 

### 7.1. Tiny Encoder channel에 따른 분류

**Phase 1: 16-Channel Baseline (V1)**
- **구조**: `1 -> 16 -> 16 -> 16`
- **결과**: 기하학적 가이드는 동작하나, 머리카락의 세밀한 결(Strands)을 표현하기엔 내부 용량이 부족함 (Blurry Results).

**Phase 2: 128-Channel "Expand-Squeeze" (V2)**
- **가설**: 내부 채널을 128로 "확장"하여 기하학적 디테일을 학습한 뒤, 16으로 압축하면 디테일이 살아날 것이다.
- **구조**: `1 -> 128 -> 128 -> 16`

**Phase 3: 256-Channel Capacity Test (V3)**
- **목적**: 128ch도 충분해 보이지만, 256ch로 확장했을 때 더 자세한 표현이 더 좋아지는지 확인.
- **구조**: `1 -> 256 -> 256 -> 16` (Overfitting 주의)

#### Architecture Comparison (Tensor Shapes)

| Step | V1 (16ch) Shape | V2 (128ch) Shape | Function |
| :--- | :--- | :--- | :--- |
| **Input** | `[B, 1, 128, 128]` | `[B, 1, 128, 128]` | **이진 마스크 입력**  |
| **Layer 1** | `[B, 16, 128, 128]` | `[B, 128, 128, 128]` | **특징 추출**  |
| **Layer 2** | `[B, 16, 128, 128]` | `[B, 128, 128, 128]` | **비선형 매핑 및 추상화** |
| **Output** | `[B, 16, 128, 128]` | `[B, 16, 128, 128]` | **latent injection** (SD3 규격으로 재압축) |

### 7.2. MM-DiT 아키텍처 통합 메커니즘

**Tiny Adapter**는 SD3.5의 **MM-DiT (Multimodal Diffusion Transformer)** 블록 내부에 복잡하게 얽히는 것이 아니라, **블록의 입력단에 Additive 방식으로 개입**합니다.

![MM-DiT Diagram](assets/mm_dit_diagram.png)

1.  **주입 위치**:
    *   위 다이어그램의 우측 상단 분홍색 노드 **$x$ (Image Latents)**가 주입 지점입니다.
    *   Tiny Adapter가 추출한 기하학적 특징($F_{mask}$)은 MM-DiT 블록 진입 직전에 입력 노이즈 $x$에 더해집니다.
    *   $$ x_{input} = x_{noise} + F_{adapter}(mask) $$

2.  **데이터 흐름**:
    *   기하학적 정보가 혼합된 $x_{input}$은 우측의 **Image Branch**를 통과합니다.
    *   **Layer Norm**과 **AdaLN (Adaptive Layer Norm)**을 거쳐 정규화 및 모듈레이션 된 후, **Joint Attention** 메커니즘의 Q, K, V로 전달됩니다.

3.  **작동 원리**:
    *   일반적인 생성 과정에서 $x$는 의미 없는 가우시안 노이즈이지만, Adapter를 통해 **기하학적 잠재 정보**가 주입됩니다.
    *   결과적으로 중앙의 **Joint Attention** 연산 과정에서, 모델은 Text+Image를 통합적으로 처리하며, 텍스트 조건 $c$ 를 렌더링할 때 우리가 주입한 "마스크 형태의 밑그림"에 강하게 **Attention**하게 됩니다.

### 7.3. 결과 비교(Train Data)

| Input Image | Mask | **16ch Adapter (V1)** | **128ch Adapter (V2)** | **256ch Adapter (V3)** |
| :---: | :---: | :---: | :---: | :---: |
| <img src="data/bald_images/01047.png" width="150"> | <img src="data/segmantic_masks/01047.png" width="150"> | <img src="results/final_hybrid/01047_result.png" width="150"> | <img src="results/final_hybrid/01047_result_v2.png" width="150"> | <img src="results/final_hybrid/01047_v3.png" width="150"> |
| <img src="data/bald_images/01056.png" width="150"> | <img src="data/segmantic_masks/01056.png" width="150"> | <img src="results/final_hybrid/01056_result.png" width="150"> | <img src="results/final_hybrid/01056_result_v2.png" width="150"> | <img src="results/final_hybrid/01056_v3.png" width="150"> |
| <img src="data/bald_images/01057.png" width="150"> | <img src="data/segmantic_masks/01057.png" width="150"> | <img src="results/final_hybrid/01057_result.png" width="150"> | <img src="results/final_hybrid/01057_result_v2.png" width="150"> | <img src="results/final_hybrid/01057_v3.png" width="150"> |

# SD 1.5 vs SD 3.5 Performance 비교(Test Data)

기존 **SD 1.5** 모델과 새로운 **SD 3.5(SONIC + TinyAdapter + Latent Blending)** 파이프라인의 헤어 생성 결과를 비교합니다.

| ID | Bald Input | Semantic Mask | SD1.5(320ch) | SD3.5(128ch) | SD3.5(256ch) |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **test1** | <img src="test_data/bald_images/test1.png" width="150"> | <img src="test_data/segmantic_masks/test1.png" width="150"> | <img src="results/v4_test_semantic/test1.png" width="150"> | <img src="results/final_hybrid/test1.png" width="150"> | <img src="results/final_hybrid/test1_v3.png" width="150"><br> |
| **test2** | <img src="test_data/bald_images/test2.png" width="150"> | <img src="test_data/segmantic_masks/test2.png" width="150"> | <img src="results/v4_test_semantic/test2.png" width="150"> | <img src="results/final_hybrid/test2.png" width="150"> | <img src="results/final_hybrid/test2_v3.png" width="150"><br> |
| **test3** | <img src="test_data/bald_images/test3.png" width="150"> | <img src="test_data/segmantic_masks/test3.png" width="150"> | <img src="results/v4_test_semantic/test3.png" width="150"> | <img src="results/final_hybrid/test3.png" width="150"> | <img src="results/final_hybrid/test3_v3.png" width="150"><br> |
| **test4** | <img src="test_data/bald_images/test4.png" width="150"> | <img src="test_data/segmantic_masks/test4.png" width="150"> | <img src="results/v4_test_semantic/test4.png" width="150"> | <img src="results/final_hybrid/test4.png" width="150"> | <img src="results/final_hybrid/test4_v3.png" width="150"><br> |
| **test5** | <img src="test_data/bald_images/test5.png" width="150"> | <img src="test_data/segmantic_masks/test5.png" width="150"> | <img src="results/v4_test_semantic/test5.png" width="150"> | <img src="results/final_hybrid/test5.png" width="150"> | <img src="results/final_hybrid/test5_v3.png" width="150"><br> |

# SD 1.5 vs SD 3.5 프롬프트 별 Performance 비교

| ID | Bald Input | Semantic Mask | Prompt | SD1.5(320ch) | SD3.5(128ch) | SD3.5(256ch) |
| :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **test1** | <img src="test_data/bald_images/test1.png" width="150"> | <img src="test_data/segmantic_masks/test1.png" width="150"> | "high quality realistic male hairstyle, low skin fade haircut, black hair, clean sides, textured top, dry matte finish, sharp hairline, natural lighting, high detail, 8k" | <img src="results/v4_test_exp/test1_1.png" width="150"> | <img src="results/final_hybrid/test1_1_v2.png" width="150"> | <img src="results/final_hybrid/test1_1.png" width="150"><br> |
| **test2** | <img src="test_data/bald_images/test1.png" width="150"> | <img src="test_data/segmantic_masks/test1.png" width="150"> | "realistic korean male two block haircut, dark brown hair, soft volume, clean contour, slightly wavy textured fringe, natural shine, high detail, 8k" | <img src="results/v4_test_exp/test1_2.png" width="150"> | <img src="results/final_hybrid/test1_2_v2.png" width="150"> | <img src="results/final_hybrid/test1_2.png" width="150"><br> |
| **test3** | <img src="test_data/bald_images/test1.png" width="150"> | <img src="test_data/segmantic_masks/test1.png" width="150"> | "realistic male textured crop haircut, short fringe, ash gray highlights on dark base, rough texture, slightly tousled top, modern barbershop style, cinematic lighting, ultra-detailed, 8k" | <img src="results/v4_test_exp/test1_3.png" width="150"> | <img src="results/final_hybrid/test1_3_v2.png" width="150"> | <img src="results/final_hybrid/test1_3.png" width="150"><br> |
| **test4** | <img src="test_data/bald_images/test1.png" width="150"> | <img src="test_data/segmantic_masks/test1.png" width="150"> | "realistic male hairstyle, natural perm with soft waves, medium length fringe pushed slightly forward, subtle brown highlights, airy movement, fluffy texture, high detail, 8k" | <img src="results/v4_test_exp/test1_4.png" width="150"> | <img src="results/final_hybrid/test1_4_v2.png" width="150"> | <img src="results/final_hybrid/test1_4.png" width="150"><br> |
| **test5** | <img src="test_data/bald_images/test1.png" width="150"> | <img src="test_data/segmantic_masks/test1.png" width="150"> | "realistic male slicked back hairstyle, blonde hair color, wet glossy texture, strong hold finish, clean sides, sharp edges, modern gentleman style, high fidelity detail, 8k" | <img src="results/v4_test_exp/test1_5.png" width="150"> | <img src="results/final_hybrid/test1_5_v2.png" width="150"> | <img src="results/final_hybrid/test1_5.png" width="150"><br> |

## 8. 학습 목표 (Training Objective)

**Native Adapter** 학습에는 Stable Diffusion 3.5의 핵심 메커니즘인 **Flow Matching** 기반의 Loss를 사용합니다.

### 8.1. Flow Matching Operator
SD 3.5는 기존 DDPM(노이즈 예측)이 아닌 **Rectified Flow** 방식을 사용합니다. 이는 데이터($x_0$)와 노이즈($x_1$) 사이를 잇는 **가장 직선적인 경로(Straight Path)**를 학습하는 것을 목표로 합니다.

*   **Forward Process (Noise Addition)**:
    $$ z_t = (1 - \sigma_t) \cdot x_0 + \sigma_t \cdot \epsilon $$
    *   $\sigma_t \in [0, 1]$: 시간 $t$에 따른 보간 비율.
    *   $t=0 \to z_0 = x_0$ (이미지)
    *   $t=1 \to z_1 = \epsilon$ (노이즈)

### 8.2. Loss Function (MSE)
모델(Transformer)은 현재 상태 $z_t$에서 **노이즈 쪽으로 향하는 속도 벡터(Velocity Field, $v$)**를 예측하도록 학습됩니다.

$$ L = || v_{pred} - v_{target} ||^2 = || \text{Transformer}(z_t, t, c) - (\epsilon - x_0) ||^2 $$

*   **$v_{target} = \epsilon - x_0$**: 원본 이미지에서 노이즈로 가는 방향 벡터.
*   **$v_{pred}$**: 모델이 예측한 방향.
*   **의미**: "TinyAdapter가 준 힌트(Mask Feature)를 참고했을 때, 모델이 **올바른 방향($\epsilon - x_0$)**으로 가고 있는가?"를 측정합니다.

즉, 우리가 사용하는 Loss는 **예측된 벡터와 실제 벡터 사이의 거리(Mean Squared Error)**입니다. 이 값이 0에 가까워질수록 모델은 마스크 힌트를 잘 이해하고 올바른 이미지를 생성하게 됩니다.

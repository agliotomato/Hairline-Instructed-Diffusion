# Seamless Hair Generation: 기술적 분석 및 방법론 

## 1. 문제 정의 

Diffusion Model기반의 헤어 생성, 특히 마스크 기반 조건부 생성환경에서 가장 중요한 문제는 생성된 헤어와 보존된 얼굴 영역 사이의 Seamlessness을 확보하는 것입니다. 일반적인 인페인팅 파이프라인에서는 주로 두 가지 아티팩트가 발생

1.  **Hard Seams (Aliasing)**: Latent 리사이징 과정에서 저차원 보간법 사용으로 인해 발생하는 계단 현상 및 거친 경계. (1024 * 1024 -> 128 * 128))
2.  **Halo/Bleed Effects**: 이진 마스킹 * nary Masking)으로 인해 생성된 텍스처가 피부톤과  자연스럽게 섞이지 못하고 색상이 번지거나 부자연스러운 경계가 형성되는 현상.

본 문서는 이러한 문제를 해결하고 고해상도의 자연스러운 헤어 통합을 달성하기 위해 적용된 기술적 방법론을 기술합니다.

## 2. 공간 resampling

### 2.1. 왜 Bilinear를 쓰면 안 되는가? 
처음에는 익숙한 **Bilinear Interpolation**을 사용하여 1024px 이미지를 처리했습니다. 하지만 실제 학습 결과물을 분석해보니 치명적인 문제가 있었습니다.
*   **Edge Definition 손실**: 리사이징 과정에서 고주파 성분(머리카락 가닥)이 뭉개졌습니다.
*   **Staircase Artifacts**: 생성된 이미지에서 머리카락 끝부분이 계단처럼 깨지는 현상이 발생했습니다. (`test6_1.png` 참고)

단순히 이미지를 줄이는 게 아니라, "어떻게 줄여야 정보를 잃지 않을까?"를 고민

### 2.2. 해결책: Lanczos Resampling 도입
신호 처리 이론을 찾아본 결과, Sinc 함수를 사용하는 **Lanczos Resampling**이 고주파 성분 보존에 유리하다는 점을 확인하고 바로 적용했습니다.
$$ L(x) = \begin{cases} \text{sinc}(x)\text{sinc}(x/a) & \text{if } -a < x < a \\ 0 & \text{otherwise} \end{cases} $$
결과는 성공적. Lanczos를 적용하자 Latent Mask의 경계가 훨씬 날카롭게 유지되었고, 확산 모델이 머리카락 끝부분을 뭉개지 않고 선명하게 생성해내기 시작했습니다.


## 3. 경계 처리

단순히 전체에 Blur를 주는 것은 정답이 아니었습니다. 블러를 많이 주면 머리카락이 풍성해지지만 디테일이 죽고, 조금 주면 디테일은 살지만 "가발"처럼 경계가 어색해졌기 때문입니다.
그래서 우리는 영역을 **두 가지(Core vs Edge)**로 나누어 다르게 처리하는 **Smart Blur** 전략을 세웠습니다.

### 3.1. 영역 분리 
핵심은 픽셀의 역할에 따라 구역을 나누는 것입니다.
1.  **Core Zone (뿌리/볼륨)**: $H_{core}$. 확실한 머리카락 영역. 여기는 노이즈가 고르게 들어가야 하니 블러를 적당히 줍니다.
2.  **Edge Zone (잔머리/경계)**: $H_{edge}$. 얼굴과 만나는 섬세한 부분. 여기는 블러를 많이 주면 얇은 선이 다 사라져버리므로, 블러를 아주 약하게 주거나 안 줘야 합니다.

### 3.2. 이중 스케일 블러링 & 마스크 합성
초기에는 Core 영역에 엄청 강한 블러(4x)를 줬었는데, 테스트해보니 오히려 머리가 "헬멧"처럼 떡지는 부작용이 있었습니다. 그래서 최종적으로는 다음과 같이 튜닝했습니다.

*   **Core Blur** (Volume): radius blur 5.0 을 그대로 사용. 부드러운 연결감을 줍니다.
*   **Light Blur** (Edge): **최대 0.8**로 제한. 잔머리 같은 High-frequency 디테일이 뭉개지지 않고 살아남도록 합니다.

$$ M_{final} = H_{core}(\text{Blur}=5.0) \oplus H_{edge}(\text{Blur}=0.8) $$
(얼굴 보호 구역 로직은 유지하여 이마 라인을 침범하지 않도록 했습니다.)

### 3.3. 마스크 강도 정규화 (Normalization)
이것이 가장 중요한 발견 중 하나였습니다.
블러를 적용하면 픽셀값(0~255)이 주변과 섞이면서 **최대 밝기(Peak Intensity)가 낮아지는 현상**이 발생합니다. (예: 255 -> 150).
값이 낮아지면 Adapter는 "여기는 머리카락을 조금만 그려"라고 해석해서, 결과물이 희미해지거나 생성이 안 되는 문제가 생깁니다.

이를 해결하기 위해 **Peak Normalization**을 추가했습니다.
$$ M_{norm} = \frac{M_{blurred}}{\max(M_{blurred})} $$
블러 후 가장 밝은 부분이 다시 **무조건 1.0(255)**이 되도록 스케일링을 해주었더니, **부드러운 경계는 유지하면서도 생성 신호는 확실하게** 줄 수 있게 되었습니다.

### 3.4. 시각화 결과 

| Before | Now |
| :---: | :---: |
| ![Smart Blur Final](debug_smart_blur_final.png) | ![V2 Final](debug_v2_final.png) |

왼쪽의 이미지는 왼쪽 옆머리 부분이 과하게 blur 처리가 되어 모델이 이 부분을 머리부분이라고 인식하지 못할 수도 있다.

#### 최종 생성 레퍼런스 비교 
이러한 마스킹 개선이 실제 생성 결과에 미치는 영향은 다음과 같습니다.

| Before | Now |
| :---: | :---: |
| ![Native2 Reference](results/native2/test4_smart.png) | ![New Pipeline Result](results/0125/test4.png) |
| **기존 결과** | **개선된 파이프라인 결과** |

#### 3.5. 최적 파라미터 구성 
위의 개선된 결과(Now)는 다음 파라미터 조합에서 최적의 성능을 보였습니다.

*   **Blur**: 0.8
*   **Adapter Scale**: 3.0 (Geometry 유도 강도 강화)
*   **Blur Radius**: 5.0 (Smart Blur 기본 반경)



## 4. 잠재 공간 일관성

### 4.1. 얼굴 색감이 왜 변할까?
Inpainting을 할 때 마스크가 없는 부분(얼굴, 배경)은 원래 그대로 나와야 한다고 생각하지만, 실제로는 VAE 인코딩/디코딩을 거치면서 **미세한 색감 변화**나 디테일 손실이 발생합니다. "분명 같은 사진인데 얼굴 톤이 묘하게 다른" 문제가 계속 발생했습니다.

### 4.2. 해결책: Latent Blending (강제 덮어쓰기)
생성 모델이 배경을 "다시 그리기"를 기대하지 않고, 그냥 수학적으로 **원본을 덮어쓰기**로 했습니다.
디노이징 스텝(Timestep $t$)이 진행될 때마다, 배경 영역($1-M$)의 Latent 값을 **원본 이미지의 Latent**로 강제로 교체합니다.
$$ z_{t-1} = M \cdot z_{pred} + (1-M) \cdot z_{bg} $$
이 한 줄의 수식을 통해 얼굴과 배경의 색감, 질감이 **원본과 동일하게 유지**되면서, 생성된 머리카락만 자연스럽게 얹어지는 결과를 얻을 수 있었습니다.

### 5.1. 프롬프트 및 모델 성능 비교
**Test Prompt**: "natural realistic male hairstyle, black hair, subtle hair detail, soft natural texture, clean look, high detail, 8k"

| ID | Semantic Mask | Before | After | Nano Banana |
| :---: | :---: | :---: | :---: | :---: |
| **test1** | ![Mask](test_data/segmantic_masks/test1.png) | ![Native2](results/native2/test1_smart.png) | ![0125](results/0125/test1.png) | ![nb](results/nano_banana/test1_nb.png) |
| **test2** | ![Mask](test_data/segmantic_masks/test2.png) | ![Native2](results/native2/test2_smart.png) | ![0125](results/0125/test2.png) | ![nb](results/nano_banana/test2_1_nb.png) |
| **test3** | ![Mask](test_data/segmantic_masks/test3.png) | ![Native2](results/native2/test3_smart.png) | ![0125](results/0125/test3.png) | ![nb](results/nano_banana/test3_nb.png) |
| **test4** | ![Mask](test_data/segmantic_masks/test4.png) | ![Native2](results/native2/test4_smart.png) | ![0125](results/0125/test4.png) | ![nb](results/nano_banana/test4_nb.png) |
| **test5** | ![Mask](test_data/segmantic_masks/test5.png) | ![Native2](results/native2/test5_smart.png) | ![0125](results/0125/test5.png) | ![nb](results/nano_banana/test5_nb.png) |
| **test6** | ![Mask](test_data/segmantic_masks/test6.png) | ![Native2](results/native2/test6_smart.png) | ![0125](results/0125/test6.png) | ![nb](results/nano_banana/test6_nano.png) |


### 5.2. 다양한 프롬프트 실험
| Prompt | test1 (Clean) | test4 (M-shape) | test6 (Complex) |
| :--- | :---: | :---: | :---: |
| "high quality realistic male hairstyle, low skin fade haircut, black hair, clean sides, textured top, dry matte finish, sharp hairline, natural lighting, high detail, 8k" | ![test1](results/0125/test1_1.png) | ![test4](results/0125/test4_1.png) | ![test6](results/0125/test6_style1_fade.png) |
| "realistic korean male two block haircut, dark brown hair, soft volume, clean contour, slightly wavy textured fringe, natural shine, high detail, 8k" | ![test1](results/0125/test1_2.png) | ![test4](results/0125/test4_2.png) | ![test6](results/0125/test6_style2_twoblock.png) |
| "realistic male textured crop haircut, short fringe, ash gray highlights on dark base, rough texture, slightly tousled top, modern barbershop style, cinematic lighting, ultra-detailed, 8k" | ![test1](results/0125/test1_3.png) | ![test4](results/0125/test4_3.png) | ![test6](results/0125/test6_style3_crop.png) |
| "realistic male hairstyle, natural perm with soft waves, medium length fringe pushed slightly forward, subtle brown highlights, airy movement, fluffy texture, high detail, 8k" | ![test1](results/0125/test1_4.png) | ![test4](results/0125/test4_4.png) | ![test6](results/0125/test6_style4_perm.png) |
| "realistic male slicked back hairstyle, blonde hair color, wet glossy texture, strong hold finish, clean sides, sharp edges, modern gentleman style, high fidelity detail, 8k" | ![test1](results/0125/test1_5.png) | ![test4](results/0125/test4_5.png) | ![test6](results/0125/test6_style5_slick.png) |

# 헤어라인 조건부 머리카락 생성 파이프라인 개요 (Hairline-Conditioned Hair Generation Pipeline)


이 파이프라인은 Stable Diffusion 기반의 접근 방식을 사용하여 대머리 얼굴 이미지에 사실적인 머리카락을 생성하며, 헤어라인 마스크로 생성 영역을 엄격하게 제어하고 텍스트 프롬프트로 스타일을 조절합니다.

1. 입력 데이터 요구사항 (Input Data Requirements)

| 구분 | 요소 | 크기/형식 | 역할 |
| :--- | :--- | :--- | :--- |
| **필수 입력** | 대머리 얼굴 이미지 (Bald Face Image) | $512 \times 512$, **RGB (3ch)** | Identity 보존 및 잠재 변수 $z$의 근원. |
| **조건 1 (구조)** | 헤어라인 / 이마 마스크 (Hairline Mask) | $512 \times 512$, **1ch** | 머리 생성 영역을 지정하고, 잠재 변수 $m$으로 변환되어 UNet에 전달. |
| **조건 2 (스타일)** | Optional Text Prompt | String | Hairstyle의 스타일(컬, 길이, 색)을 Stable Diffusion의 cross-attention으로 처리. |


2. Latent Encoding (Stage 1) 및 UNet 입력 구성

| 단계 | 원본 | 잠재 변수 | 크기 (Resolution) | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **1. Bald Image Encoding** | Bald Image | $z$ | $4 \times 64 \times 64$ | VAE Encoder 사용. |
| **2. Mask Downsampling** | Mask Image | $m$ | $1 \times 64 \times 64$ | $512 \rightarrow 64$ 해상도로 축소. |
| **3. Latent Concatenation** | $z, m$ | $\text{input\_latent}$ | **$5 \times 64 \times 64$** | **UNet의 5채널 입력**으로 사용. |

3. Stage 2: 확산 UNet 순방향 전파 (Diffusion UNet Forward) (노이즈 예측)핵심 확산 프로세스는 공간적 조건을 수용하도록 수정됩니다.표준 확산 순방향 전파:$$\epsilon_\theta = \text{UNet}(x_t, t, \text{cond})$$여기서 $x_t$는 노이즈가 추가된 잠재 변수이며 $\epsilon_\theta$는 예측된 노이즈입니다.수정된 UNet 입력:입력 잠재 변수 채널: 5 채널 ($z$: 4ch + $m$: 1ch).조건 ($\text{cond}$): 선택적 텍스트 프롬프트 임베딩 (교차 어텐션을 통해 처리).헤어라인 조건: 마스크 정보는 잠재 공간에서 채널 결합을 통해 직접 전달됩니다.UNet의 역할: UNet은 마스크의 공간적 패턴을 활용하여 지정된 영역 내에서만 머리카락을 생성하도록 학습합니다.

4. Stage 3: 

확산 학습 단계 (Diffusion Training Step)
학습 목표는 마스크 및 텍스트 조건을 통합한 표준 노이즈 예측 손실(loss)을 기반으로 합니다.


1. 입력 준비: 대머리 이미지 $\rightarrow$ $z$. 마스크 $\rightarrow$ $m$. $\text{input\_latent} = \text{concat}([z, m])
2. 노이즈 추가: 타겟 머리카락 잠재 변수에 실제 노이즈 $\epsilon$을 추가하여 $x_t$ (노이즈 잠재 변수)를 얻습니다.
3. 노이즈 예측: $\epsilon_\theta = \text{UNet}(x_t, t, \text{text\_cond})
4. 손실 계산: 평균 제곱 오차 (MSE) 노이즈 예측 손실: $\text{Loss} = ||\epsilon - \epsilon_\theta||^2
5. 최적화: 역전파(Backpropagation) 및 UNet 업데이트.(향후 개선: 얼굴 구조를 더욱 안정화하기 위해 정체성(Identity) 손실 또는 지각(Perceptual) 손실을 추가할 수 있습니다.)

5. Stage 4: 추론 / 샘플링 (Inference / Sampling)

머리카락 생성 프로세스는 역확산(denoising) 방식을 사용하여 최종 이미지를 생성합니다.
1. 입력: 대머리 이미지, 헤어라인 마스크, 선택적 텍스트 프롬프트.
2. 잠재 인코딩: 입력을 인코딩하고 결합하여 $\text{input\_latent}$ **($5 \times 64 \times 64$)**를 형성합니다.
3. 역확산 실행: DDIM 또는 PLMS 기반의 역 프로세스를 실행합니다:$$x_{t-1} = \text{denoise\_step}(\text{UNet}(\dots))$$
이 과정은 $\text{input_latent}$와 $\text{text_cond}$의 안내에 따라 초기 무작위 노이즈를 반복적으로 제거하여 최종 잠재 변수에 도달합니다.
4. 디코딩: 최종 잠재 변수를 **VAE 디코더 (Decoder)**로 변환합니다.
5. 출력: $\rightarrow$ 머리카락이 생성된 3채널 $512 \times 512$ RGB 이미지.


6. 출력 특징 (Output Features)정체성 보존: 생성된 머리카락이 원본 얼굴 구조에 자연스럽게 통합됩니다.정확한 공간 제어: 머리카락 생성은 헤어라인 마스크로 지정된 영역 내로 제한됩니다.스타일 사용자 정의: 텍스트 프롬프트를 통해 헤어스타일의 미적 속성을 유연하게 제어할 수 있습니다.
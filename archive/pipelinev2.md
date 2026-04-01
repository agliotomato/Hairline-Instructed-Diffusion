이 모델의 Training pipeline은 다음 개념을 반드시 따라야 합니다:

1. Training Objective

Diffusion의 x₀ = z_orig 구조를 사용한다.

즉, **원본 머리 이미지(I_orig)**를 VAE Encoder로 latent(z_orig)로 변환한 것이
Diffusion이 복원해야 하는 **정답(x₀)**이다.

반대로, **대머리 이미지(I_bald)**는 x₀가 아니다.
이건 정답이 아니라 **조건(cond)**이다.

모델이 배워야 하는 함수는 다음과 같다:

(bald image, hairline mask, text prompt) -> original hair latent


이 구조를 기반으로 학습해야
추론 시 어느 텍스트/헤어라인 mask가 들어와도
플라우저블한(realistic) 머리카락을 생성할 수 있다.

2. 입력 데이터 구조

Training 데이터는 다음 3개가 한 세트다:

I_orig : 원본 얼굴 + 머리 이미지 (512x512 RGB)

I_bald : 동일한 사람을 bald-converter로 만든 대머리 이미지

M : hairline mask (512x512 1ch)

여기서 I_orig만이 “정답(target)”이다.

3. Latent Encoding

z_orig = VAE.encode(I_orig) → (4, 64, 64)

z_bald = VAE.encode(I_bald) → (4, 64, 64)

m = downsample(mask) → (1, 64, 64)

4. UNet 입력 (in_channels=5)

UNet은 기본 Stable Diffusion UNet을 수정하여
in_channels = 5로 만든다:

4채널: noisy target latent xₜ

1채널: hairline mask latent (m)

즉,

unet_input = concat(x_t, m)  # shape: (5, 64, 64)

5. Condition (encoder_hidden_states)

cross-attention의 조건(cond)은 다음 구조여야 한다:

mask_token

m(1×64×64)을 global pooling → Linear → (1, D)

(옵션) bald_token

z_bald를 global pooling → Linear → (1, D)

필요 시 identity 안정화를 위해 사용

text_tokens

CLIP text encoder 출력

최종 cond:

cond = [mask_token, text_tokens]
# 또는 cond = [bald_token, mask_token, text_tokens]


UNet forward:

noise_pred = unet(unet_input, t, encoder_hidden_states=cond)

6. Diffusion Training Step

훈련 루프는 standard noise prediction loss:

x_t = sqrt(alpha_t)*z_orig + sqrt(1-alpha_t)*epsilon  # epsilon ~ N(0, I)

epsilon_pred = UNet([x_t, m], t, cond)

loss = MSE(epsilon, epsilon_pred)

7. Inference Pipeline

추론 시 입력은:

사용자의 대머리 이미지 I_bald_input

의사가 그린 hairline mask M_doctor

(선택) text prompt

이것을 encode하여:

z_bald_test = E(I_bald_input)
m_test = downsample(M_doctor)


초기 노이즈 x_T에서 DDIM/DDPM reverse 과정을 실행해
최종 x₀를 얻고,
Decoder로 이미지를 복원한다.

8. 요구사항 요약 (Codex가 반드시 지켜야 하는 사항)

x₀는 z_orig로 학습한다 (원본 머리 latent).

UNet 입력은 5채널(x_t + mask map).

hairline mask는 UNet 입력 + cross-attention token 두 방식으로 모두 사용된다.

대머리 latent(z_bald)는 cond로만 사용하며 target이 아니다.

Inference에서는 의사가 그린 mask를 그대로 넣는다.

standard DDPM/DDIM scheduler 사용.

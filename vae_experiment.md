# VAE 재구성 실험 (VAE Reconstruction Experiment)

## 1. 목표 (Goal)
Stable Diffusion v1.5에서 사용하는 기본 VAE(kl-f8 기반 AutoencoderKL)가 **뾰족한 머리카락(spiky hair)**, **날카로운 엣지(edge)**, **기타 고주파(High-Frequency) 세부 텍스처**를 얼마나 잘 보존하는지 평가한다.

이를 위해 원본 이미지와 VAE 재구성 이미지를 비교하여, 다음을 구분하는 것이 목적이다:
- VAE 단계에서 이미 디테일이 사라지는지
- 아니면 VAE는 충분히 살리는데 이후 Diffusion/UNet 단계에서 문제가 생기는지

## 2. 변경 사항 (Proposed Changes)

### (A) 신규 스크립트 추가
- **파일 경로**: `experiment_vae_reconstruction.py` (루트 디렉토리)
- **기능 요약**:
    - Stable Diffusion v1.5의 VAE(AutoencoderKL)를 로드
    - 입력 이미지 디렉토리를 순회하며 **[원본 → VAE 인코딩 → 디코딩 → 재구성 이미지]**를 생성
    - [원본 | 재구성] 형태의 그리드 이미지를 저장

### (B) 모델 로드 (Model Loading)
- **사용 모델**: `runwayml/stable-diffusion-v1-5`의 VAE (AutoencoderKL) 또는 로컬 경로
- **동작**: `StableDiffusionPipeline` 대신 VAE만 단독 로드하거나 `pipe.vae` 사용

### (C) 함수 설계: `reconstruct_image(image_path, vae, device)`
- **입력**:
    - `image_path`: 원본 이미지 경로
    - `vae`: 로드된 AutoencoderKL 인스턴스
    - `device`: "cuda" 또는 "cpu"
- **동작**:
    - 이미지를 로드 (PIL)
    - 512x512 기준으로 resize/center crop
    - `torch.Tensor`로 변환 및 정규화 ([-1, 1] 범위)
    - `vae.encode()`로 latent 추출 (z)
    - `vae.decode(z)`로 다시 이미지를 복원
    - 복원 이미지를 PIL.Image로 변환하여 반환
- **출력**: `(original_pil, reconstructed_pil)` 또는 두 이미지를 합친 그리드 이미지

### (D) 메인 루프 (Main Loop)
- **대상 디렉토리**: `input_dir` 내의 이미지들(.png, .jpg 등)
- **동작**:
    - 각 이미지에 대해 `reconstruct_image(...)` 호출
    - [원본 | 재구성] 형태의 가로 그리드 이미지 생성
    - `outputs/vae_reconstruction/` 아래에 `originalname_recon.png` 형태로 저장

## 3. 검증 계획 (Verification Plan)

### (A) 자동화 검증 (Automated Test)
- 샘플 이미지 세트에 대해 스크립트 실행:
  ```bash
  python experiment_vae_reconstruction.py --input_dir data/test_images
  ```
- 스크립트 정상 종료 및 GPU/CPU 동작 확인

### (B) 수동 시각 검증 (Manual Visual Check)
- 저장된 [원본 | 재구성] 이미지를 육안으로 확인하며 다음을 중점적으로 평가:
    - 헤어라인 뾰족함 유지 여부
    - 머리카락 끝(edge), M자형/V자형 헤어라인의 선명도
    - **윤곽선(edge)**의 뭉개짐 또는 번짐 정도
    - 배경/얼굴 대비, 노이즈, 블러 정도

### 결과 해석
- **고주파 디테일 보존 시**: VAE는 충분함 → 문제는 Diffusion/UNet/Conditioning 단계
- **VAE에서 뾰족함 소실 시**: VAE 구조/압축 손실이 원인 → VAE 튜닝 또는 별도 인코더 필요

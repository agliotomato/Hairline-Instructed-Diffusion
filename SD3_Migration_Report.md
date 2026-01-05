# SD3.5 Hybrid ControlNet Migration Report

## 1. 개요 (Overview)
본 문서는 Stable Diffusion 3.5 Medium 모델을 기반으로 한 **Hybrid Dual-Stream ControlNet** (Geometry + Identity) 학습 시스템 구축 과정과 그 결과를 정리합니다.

## 2. 주요 구현 사항 (Achievements)

### 2.1. Hybrid ControlNet 아키텍처 구현
기존 SD 1.5/SDXL 기반의 파이프라인을 SD3.5의 **MMDiT (Multimodal Diffusion Transformer)** 구조에 맞게 성공적으로 이식했습니다.

*   **Dual-Stream 구조**:
    *   **Stream A (Geometry)**: 1-Channel Semantic Mask 입력을 처리. `SD3ControlNetModel`의 `extra_conditioning_channels=1` 설정으로 초기화.
    *   **Stream B (Identity)**: 16-Channel VAE Latent (Masked Bald Image) 입력을 처리. `SD3ControlNetModel`을 수동 설정하여 `extra_conditioning_channels=16`으로 초기화.
*   **Input Concatenation**: SD3 ControlNet의 요구사항에 맞춰 `Noisy Latents(16ch)`와 `Condition(1ch or 16ch)`을 채널 차원에서 결합하여 모델에 전달하는 로직 구현.

### 2.2. Training Script & Logic
*   **Rectified Flow Loss**: SD3의 Flow Matching 학습 로직을 수동으로 구현 (Noise 예측이 아닌 Velocity 예측).
*   **Scheduler Logic**: `scheduler.sigmas`를 활용한 Noise 추가 및 Timestep 샘플링 로직 구현.
*   **Mixed Precision**: FP16 학습을 위한 데이터 캐스팅 및 Autocast 적용 완료.

### 2.3. VRAM 최적화 시도 (Aggressive Optimization)
A100 (40GB) 환경에서 거대한 SD3.5 모델을 구동하기 위해 극한의 최적화를 시도했습니다.

1.  **Gradient Checkpointing**:
    *   ControlNet A, B 뿐만 아니라 **Frozen Transformer**에도 Checkpointing을 적용하여 Activation 메모리 절약.
2.  **On-Demand Offloading (Transformer Shuttling)**:
    *   **T5 Text Encoder**와 **Transformer**가 VRAM에 동시에 존재하지 않도록 제어.
    *   인코딩 시: T5 GPU 로드 -> 인코딩 -> 즉시 CPU 방출.
    *   학습 시: Transformer GPU 로드 -> Forward -> 즉시 CPU 방출.
3.  **Aggressive Cache Cleanup**: 각 단계마다 `torch.cuda.empty_cache()`를 호출하여 파편화된 메모리 즉시 반환.

## 3. 한계점 및 중단 원인 (Limitations & Blockers)

### 3.1. 하드웨어 제약 (Hardware Constraints)
**결론: A100 40GB VRAM으로는 SD3.5 Medium 학습(특히 T5 XXL 인코더 포함)이 불가능에 가까움.**

*   **T5 Text Encoder**: SD3.5의 성능 핵심인 T5 XXL 모델 자체가 매우 거대하여(fp16 기준 약 10GB+), 이를 로드하는 순간 학습을 위한 여유 공간이 거의 남지 않음.
*   **Transformer & ControlNet**: 기본 Transformer(2.5B)와 ControlNet 2개(각각 2.5B Copy)가 합쳐지면 모델 가중치만으로도 VRAM을 상당히 점유함.
*   **Activation Memory**: 해상도 1024x1024 학습 시, Gradient Checkpointing을 써도 Backpropagation을 위한 중간 값 저장에 막대한 메모리가 필요함.

### 3.2. 실행 결과
*   Transformer와 T5를 교대로 올리는 'Shuttling' 전략까지 썼음에도 불구하고, **OOM (Out Of Memory)** 발생.
*   배치 사이즈 1로도 메모리 부족.

## 4. 향후 권장 사항 (Recommendations)

### 4.1. 하드웨어 업그레이드
*   **H100 (80GB)**: 80GB VRAM에서는 현재 설정(배치 1)으로 구동 가능성이 매우 높음.
*   **Multi-GPU**: A100 4개 이상을 사용하여 DeepSpeed Stage 2/3 (Model Parallelism / ZeRO)를 적용하면 메모리 문제를 해결할 수 있음.

### 4.2. 경량화 전략
*   **T5 Quantization**: T5 인코더를 8-bit (`bitsandbytes`)로 로드하면 메모리를 약 50% 절약 가능. (가장 현실적인 해결책)
*   **Pre-computed Embeddings**: 학습 전에 모든 데이터셋의 텍스트 임베딩을 미리 계산하여 디스크에 저장해두고, 학습 시에는 T5를 아예 로드하지 않는 방식. (스토리지 공간 필요하지만 VRAM 대폭 절약)

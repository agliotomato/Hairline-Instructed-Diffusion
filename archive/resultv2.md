# VERSION 2

`pipelinev2.md`에 학습 파이프라인 개념 정리. 아래는 Lambda 기준 실행 스니펫과 산출물 위치.

## 1. Training — train_hairline_cond_v2.py

필수 입력: `data/original_images`(정답 x₀), `data/bald_images`(조건), `data/only_forehead_line`(마스크).

```bash
python3 train_hairline_cond_v2.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --orig_dir data/original_images \
  --bald_dir data/bald_images \
  --mask_dir data/only_forehead_line \
  --output_dir runs/hairline_cond_v2_nomask \
  --train_batch_size 2 \
  --num_train_epochs 10 \
  --learning_rate 1e-5 \
  --checkpointing_steps 1000 \
  --mixed_precision fp16 \
  --dataloader_num_workers 0
```

산출물: `runs/hairline_cond_v2_nomask/` (UNet 가중치 + conditioner 토큰).
주요 인자 설명: `checkpointing_steps`(모델 저장 주기), `mixed_precision`(fp16로 메모리/속도 최적화), `train_batch_size`·`num_train_epochs`·`learning_rate`(표준 학습 하이퍼파라미터).

## 2. Inference — infer_hairline_cond_v2.py

입력: 추론용 대머리 이미지, 의사/사용자 마스크, (옵션) 프롬프트. 마스크 소스에 따라 두 패턴을 사용했다.


```bash
python3 infer_hairline_cond_v2.py \
  --model_dir runs/hairline_cond_v2_nomask/unet \
  --conditioner_path runs/hairline_cond_v2_nomask/conditioner.pt \
  --bald_path data/bald_images/01047.png \
  --mask_path data/forehead_line/01047.png \
  --num_inference_steps 30 \
  --guidance_scale 5.0 \
  --num_samples 2 \
  --seed 42 \
  --out_dir output/v2_samples \
  --dtype fp16 \
  --device cuda
```

주요 인자 설명: `num_inference_steps`(디노이징 스텝 수), `guidance_scale`(클래스프리 가이던스 세기), `num_samples`(한 번에 생성할 개수), `seed`(재현성), `dtype`(fp16 추론), `device`(cuda/auto 선택).


샘플 결과: `result/v2_samples/sample_000.png`, `sample_001.png`.

<div align="center">
  <img src="data/original_images/01047.png" alt="original_input" 
  width="18% "/ >
  <img src="data/bald_images/01047.png" alt="bald_input" width="18%" />
  <img src="data/forehead_line/01047.png" alt="mask_input_A" width="18%" />
  <img src="result/v2_samples/0.png" alt="sample_000" width="18%" />
  <img src="result/v2_samples/1.png" alt="sample_001" width="18%" />
</div>

**(B) only_forehead_line 사용 (마스크 경로만 변경)**
샘플 결과: `result/v2_samples/only_forehead_line01047-1.png`, `only_forehead_line01047-2.png`.

주요 인자 설명: `num_inference_steps`(디노이징 스텝 수), `guidance_scale`(클래스프리 가이던스 세기), `num_samples`(한 번에 생성할 개수), `seed`(재현성), `dtype`(fp16 추론), `device`(cuda/auto 선택).

<div align="center">
  <img src="data/original_images/01047.png" alt="bald_input" width="18%" />
  <img src="data/bald_images/01047.png" alt="bald_input" width="18%" />
  <img src="data/only_forehead_line/01047.png" alt="mask_input_B" width="18%" />
  <img src="result/v2_samples/only_forehead_line01047-1.png" alt="sample_1" width="18%" />
  <img src="result/v2_samples/only_forehead_line01047-2.png" alt="sample_2" width="18%" />
</div>


## 로그를 기준으로 각 파일/결과물이 하는 일 정리.

### 한 줄 요약
1. 텍스트 쪽(기본 SD v1-5에서 가져온 것)
- `tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`
- `config.json`, `model.safetensors` (CLIP text encoder 쪽)

2. UNet(harirline_cond_v2_nomask 결과물)
- `runs/hairline_cond_v2_nomask/unet/config.json`
-  `runs/hairline_cond_v2_nomask/unet/diffusion_pytorch_model.safetensors`

3. Conditioner (학습 결과)
- `runs/hairline_cond_v2_nomask/conditioner.pt`

4. 스커쥴러 & 샘플링 설정
- `scheduler_config.json`

### 로그 핵심
- 실행 커맨드(only_forehead_line): `python3 infer_hairline_cond_v2.py --model_dir runs/hairline_cond_v2_nomask/unet --conditioner_path runs/hairline_cond_v2_nomask/conditioner.pt --bald_path data/bald_images/01047.png --mask_path data/only_forehead_line/01047.png --num_inference_steps 30 --guidance_scale 5.0 --num_samples 2 --seed 42 --out_dir output/v2_samples --dtype fp16 --device cuda`

### UNet
`runs/hairline_cond_v2_nomask/unet/config.json`
`runs/hairline_cond_v2_nomask/unet/diffusion_pytorch_model.safetensors`

config.json: 100% ...
diffusion_pytorch_model.safetensors: 100%|████| 335M/335M

- `config.json`
    - UNet 구조 정보(채널 수, 블록 구성, attention head 수, in/out 채널 등)이 들어 있음
    - hairline 조건을 어떻게 주입했는지(입력 채널 5로 수정)에 대한 정보도 포함되어 있음

- `diffusion_pytorch_model.safetensors`
    - train_hairline_cond_v2.py로 학습시킨 finetuned Unet 가중치
    - Bald Image latent + hairline mask를 받아서 각 스텝에서 노이즈 예측을 출력하는 핵심 네트워크임

### Conditioner
- Bald 이미지 + hairline mask를 받아 UNet의 extra conditioning tensor를 만들어주는 모듈
- inference에서의 역할 : UNet forward에 들어가서 hairline 형태를 반영한 노이즈 예측이 이루어짐.

### 사전학습/다운로드(backbone) 구성요소
- CLIP 텍스트 인코더 및 토크나이저: `tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`, `model.safetensors`, `config.json` (Stable Diffusion v1-5에서 그대로 사용).
- (필요 시) VAE 등 다른 SD 기본 컴포넌트도 동일하게 사전학습 가중치를 사용.

- guidance : classifier-free guidance로, 무조건 예측과 조건 예측 차이를 키워 조건을 더 강하게 따르게 한다. 값이 커질 수록 텍스트/조건을 더 따르지만, 노이즈와 분산도 커져, ID나 질감이 쉽게 뒤틀린다. 

### 학습 재개/디버깅용 (추론에는 미사용)
- `checkpoint-*/model*.safetensors`, `optimizer.bin`, `scheduler.bin`, `scaler.pt`, `random_states_*.pkl` 등 중간 체크포인트: 학습 재시작/디버깅용.

### guidance 5.0 vs guidance 1.0 

<div align="center">
  <img src="result/v2_samples/only_forehead_line01047-1.png" width="18%" />
  <img src="result/v2_samples/only_forehead_line01047-2.png" width="18%" />
  <img src="output/v2_samples/sample_000.png" width="18%" />
  <img src="output/v2_samples/sample_001.png" width="18%" />
  
</div>

guidance 영향은 미미

### 프롬프트 추가 버전.
```bash
 python3 infer_hairline_cond_v2.py \
    --model_dir runs/hairline_cond_v2_nomask/unet \
    --conditioner_path runs/hairline_cond_v2_nomask/conditioner.pt \
    --bald_path data/bald_images/01047.png \
    --mask_path data/only_forehead_line/01047.png \
    --num_inference_steps 30 \
    --guidance_scale 2.5 \
    --num_samples 5 \
    --seed 42 \
    --out_dir output/v2_samples_prompt_male \
    --dtype fp16 \
    --device cuda \
    --prompt "male portrait, natural studio lighting, neutral expression, realistic skin, soft light" \
    --negative_prompt "blur, distortion, extra hair, artifacts, low quality"
```
<div align="center">
  <img src="output/v2_samples_prompt_male/sample_000.png" width="18%" />
  <img src="output/v2_samples_prompt_male/sample_001.png" width="18%" />
  <img src="output/v2_samples_prompt_male/sample_002.png" width="18%" />
  <img src="output/v2_samples_prompt_male/sample_003.png" width="18%" />
  <img src="output/v2_samples_prompt_male/sample_004.png" width="18%" />
</div>

### 현재의 문제점

- 초기 상태가 랜덤 노이즈임 -> identity가 보존이 안됨
- 랜덤 노이즈에는 얼굴 ID에 해당하는 정보가 단 1비트도 없음.



### 해결 방안: Latent Initialization (Identity-Preserving Init) <<< >>

- **아이디어**: 초기 Latent를 순수 노이즈($\epsilon$)가 아닌, **대머리 이미지의 Latent($z_{bald}$)에 노이즈를 섞은 상태**로 시작한다.
- **수식**: $x_T = z_{bald} + \sigma \cdot \epsilon$
- **효과**: 초기 상태부터 얼굴의 구조적 정보(눈, 코, 입, 얼굴형)를 강력하게 가지고 시작하므로, Diffusion 과정에서 Identity가 무너지는 것을 방지한다.

### 결과물 (2025-11-25 실험)
- **설정**: `init_latent="zbald"`, `noise_strength=0.8`
- **파일명**: `sample_20251125_030828_*.png`

<div align="center">
  <img src="output/v2_samples_prompt_male/sample_20251125_030828_000.png" width="18%" />
  <img src="output/v2_samples_prompt_male/sample_20251125_030828_001.png" width="18%" />
  <img src="output/v2_samples_prompt_male/sample_20251125_030828_002.png" width="18%" />
  <img src="output/v2_samples_prompt_male/sample_20251125_030828_003.png" width="18%" />
  <img src="output/v2_samples_prompt_male/sample_20251125_030828_004.png" width="18%" />
</div>

안하는 걸로. 그냥 discard 할게염.

# 개선 사항 및 학습 정리

## 1. 더 날카로운 Hair-Semantic Mask 생성 (SegFace)
기존 마스킹 방식의 한계를 극복하기 위해 **SegFace**를 도입하였습니다.
- **목적**: 마스크의 경계면(Edge)을 더욱 날카롭고 명확하게 생성하여, 헤어 생성 시 경계가 흐려지는 문제를 해결합니다.
- **내용**: SegFace 모델을 활용하여 머리카락(Hair), 얼굴(Face), 배경(Background)을 정밀하게 분리하고, 이를 학습 Conditioning에 활용함으로써 생성된 이미지의 디테일을 향상시켰습니다.

### Mask Quality Comparison
| 원본 이미지 (Original) | BiSeNet | SegFace |
| :---: | :---: | :---: |
| ![Original](data/original_images/01047.png) | ![BiSeNet](data/segmantic_masks/01047_c.png) | ![SegFace](data/segmantic_masks/01047.png) |
| ![Original](data/original_images/02518.png) | ![BiSeNet](data/segmantic_masks/02518_c.png) | ![SegFace](data/segmantic_masks/02518.png) |
| ![Original](data/original_images/04731.png) | ![BiSeNet](data/segmantic_masks/04731_c.png) | ![SegFace](data/segmantic_masks/04731.png) |

## 2. SD3.5 해상도에 맞춘 재학습 (계단 현상 완화)
SD3.5 모델의 Native Resolution에 맞춰 학습 해상도를 조정하였습니다.
- **문제점**: 기존 저해상도 학습 또는 해상도 불일치로 인해 생성된 이미지의 헤어라인이나 외곽선에서 **계단 현상(Aliasing)**이 발생하는 문제가 있었습니다.
- **개선**: 학습 데이터와 모델 입력을 SD3.5의 최적 해상도에 맞추어 재학습을 진행하였으며, 이를 통해 곡선과 사선이 훨씬 부드럽고 자연스럽게 표현되기를 기대

### smart_blur (이마라인 보호 + 외곽 강블러)
`test_sd3_native_adapter_inference.py`의 `--smart_blur` 옵션은 **이마/헤어라인은 약블러로 유지**하고 **이마라인이 아닌 외곽 영역은 더 강하게 블러**하여 계단 현상을 완화하는 용도입니다.
- **강블러(외곽 볼륨)**: `heavy_radius = blur_radius * 4.0`로 가우시안 블러를 크게 적용해 외곽을 부드럽게 처리
- **약블러(헤어라인)**: 기본 `blur_radius`만 적용해 헤어라인 디테일을 유지
- **보호 영역 생성**: 얼굴 마스크를 `MaxFilter(41)`로 팽창시켜 헤어라인 근처를 포함시키고, `GaussianBlur(15)`로 완만한 전이 마스크 생성
- **합성 로직**: 보호 영역(이마/헤어라인 근처)은 약블러, 바깥쪽은 강블러를 사용하도록 `Image.composite`로 혼합

## 3. 학습 원리 및 정리
현재 적용된 주요 학습 원리는 다음과 같습니다.
- **Native Adapter / TinyAdapter**: 대규모 모델인 SD3.5 전체를 파인튜닝하는 대신, 경량화된 Adapter 모듈을 추가하여 효율적으로 Hair Style을 제어합니다.
- **Mask-Guidance**: SegFace로 생성된 고품질 마스크를 가이드로 사용하여, 변경이 필요한 헤어 영역에만 집중적으로 노이즈를 주입하고 복원하는 방식을 취합니다.
- **High-Resolution Tuning**: 고해상도 입력을 처리할 수 있도록 파이프라인을 최적화하여 텍스처 품질을 유지합니다.

## 4. 비교 표 (SD1.5 vs Before vs Now)

| 구분 | Semantic Mask | SD 1.5 | Before | Now |
| :---: | :---: | :---: | :---: | :---: |
| **test1** | <img src="test_data/segmantic_masks/test1.png" width="256"/> | <img src="results/v4_test_semantic/test1.png" width="256"/> | <img src="results/final_hybrid/test1.png" width="256"/> | <img src="results/native2/test1_smart.png" width="256"/> |
| **test2** | <img src="test_data/segmantic_masks/test2.png" width="256"/> | [!NoData](<img src="results/v4_test_semantic/test2.png" width="256"/>) | [!NoData](<img src="results/final_hybrid/test2.png" width="256"/>) | <img src="results/native2/test2_smart.png" width="256"/> |
| **test3** | <img src="test_data/segmantic_masks/test3.png" width="256"/> | <img src="results/v4_test_semantic/test3.png" width="256"/> | <img src="results/final_hybrid/test3.png" width="256"/> | <img src="results/native2/test3_smart.png" width="256"/> |
| **test4** | <img src="test_data/segmantic_masks/test4.png" width="256"/> | <img src="results/v4_test_semantic/test4.png" width="256"/> | <img src="results/final_hybrid/test4.png" width="256"/> | <img src="results/native2/test4.png" width="256"/> |
| **test5** | <img src="test_data/segmantic_masks/test5.png" width="256"/> | <img src="results/v4_test_semantic/test5.png" width="256"/> | <img src="results/final_hybrid/test5.png" width="256"/> | <img src="results/native2/test5_result.png" width="256"/> |
| **test6** | <img src="test_data/segmantic_masks/test6.png" width="256"/> | [!NoData](<img src="results/v4_test_semantic/test6.png" width="256"/>) | [!NoData](<img src="results/final_hybrid/test6.png" width="256"/> )| <img src="results/native2/test6_smart.png" width="256"/> |

### 프롬프트 대응능력 비교(adapter 1.5, mask_dilation 10)

| ID | Semantic Mask | Prompt | SD1.5 | Before | Now |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **test1** | <img src="test_data/segmantic_masks/test1.png" width="150"> | "high quality realistic male hairstyle, low skin fade haircut, black hair, clean sides, textured top, dry matte finish, sharp hairline, natural lighting, high detail, 8k" | <img src="results/v4_test_exp/test1_1.png" width="150"> | <img src="results/final_hybrid/test1_1_v2.png" width="150"> | <img src="results/native2/test1_1.png" width="150"> |
| **test2** | <img src="test_data/segmantic_masks/test1.png" width="150"> | "realistic korean male two block haircut, dark brown hair, soft volume, clean contour, slightly wavy textured fringe, natural shine, high detail, 8k" | <img src="results/v4_test_exp/test1_2.png" width="150"> | <img src="results/final_hybrid/test1_2_v2.png" width="150"> | <img src="results/native2/test1_2.png" width="150"> |
| **test3** | <img src="test_data/segmantic_masks/test1.png" width="150"> | "realistic male textured crop haircut, short fringe, ash gray highlights on dark base, rough texture, slightly tousled top, modern barbershop style, cinematic lighting, ultra-detailed, 8k" | <img src="results/v4_test_exp/test1_3.png" width="150"> | <img src="results/final_hybrid/test1_3_v2.png" width="150"> | <img src="results/native2/test1_3.png" width="150"> |
| **test4** | <img src="test_data/segmantic_masks/test1.png" width="150"> | "realistic male hairstyle, natural perm with soft waves, medium length fringe pushed slightly forward, subtle brown highlights, airy movement, fluffy texture, high detail, 8k" | <img src="results/v4_test_exp/test1_4.png" width="150"> | <img src="results/final_hybrid/test1_4_v2.png" width="150"> | <img src="results/native2/test1_4.png" width="150"> |
| **test5** | <img src="test_data/segmantic_masks/test1.png" width="150"> | "realistic male slicked back hairstyle, blonde hair color, wet glossy texture, strong hold finish, clean sides, sharp edges, modern gentleman style, high fidelity detail, 8k" | <img src="results/v4_test_exp/test1_5.png" width="150"> | <img src="results/final_hybrid/test1_5_v2.png" width="150"> | <img src="results/native2/test1_5.png" width="150"> |

### 추가 데이터 다양한 스타일 실험(adapter 1.5, mask_dilation 10)
| Prompt | test2_semantic | test2_result | test6_semantic | test6_result |
| :--- | :---: | :---: | :---: | :---: |
| "high quality realistic male hairstyle, low skin fade haircut, black hair, clean sides, textured top, dry matte finish, sharp hairline, natural lighting, high detail, 8k" | <img src="test_data/segmantic_masks/test2.png" width="150"> | <img src="results/native2/test2_1.png" width="150"> | <img src="test_data/segmantic_masks/test6.png" width="150"> | <img src="results/native2/test6_1.png" width="150"> |
| "realistic korean male two block haircut, dark brown hair, soft volume, clean contour, slightly wavy textured fringe, natural shine, high detail, 8k" | <img src="test_data/segmantic_masks/test2.png" width="150"> | <img src="results/native2/test2_2.png" width="150"> | <img src="test_data/segmantic_masks/test6.png" width="150"> | <img src="results/native2/test6_2.png" width="150"> |
| "realistic male textured crop haircut, short fringe, ash gray highlights on dark base, rough texture, slightly tousled top, modern barbershop style, cinematic lighting, ultra-detailed, 8k" | <img src="test_data/segmantic_masks/test2.png" width="150"> | <img src="results/native2/test2_3.png" width="150"> | <img src="test_data/segmantic_masks/test6.png" width="150"> | <img src="results/native2/test6_3.png" width="150"> |
| "realistic male hairstyle, natural perm with soft waves, medium length fringe pushed slightly forward, subtle brown highlights, airy movement, fluffy texture, high detail, 8k" | <img src="test_data/segmantic_masks/test2.png" width="150"> | <img src="results/native2/test2_4.png" width="150"> | <img src="test_data/segmantic_masks/test6.png" width="150"> | <img src="results/native2/test6_4.png" width="150"> |
| "realistic male slicked back hairstyle, blonde hair color, wet glossy texture, strong hold finish, clean sides, sharp edges, modern gentleman style, high fidelity detail, 8k" | <img src="test_data/segmantic_masks/test2.png" width="150"> | <img src="results/native2/test2_5.png" width="150"> | <img src="test_data/segmantic_masks/test6.png" width="150"> | <img src="results/native2/test6_5.png" width="150"> |

## 느낀 점 정리
1. 이마라인은 비교적 잘 나온다.
2. 계단 현상이 주요 문제다. 블러를 세게 줘도 효과가 미미했고, 이마라인을 엄격하게 맞추기 위해 어댑터 강도를 높이면 머리 끝부분의 계단 현상이 더 심해지는 것으로 보인다.
그 trade-off를 해결할 수 있는 방안? 최적의 어댑터 강도를 찾을 수 있을까?(1.0 ~ 1.5 적당한 것으로 보임)
3. 짧은 머리는 잘 나오는데, 데이터 대부분이 짧은 머리라 그 패턴을 더 잘 학습한 영향일 가능성이 있다. 긴 머리는 퀄리티가 비교적 떨어지므로 긴 머리 데이터 보강이 필요할 수 있다.
4. 머리가 잘리는 부분은 dilation으로 보정이 가능하다. 다만 결과적으로 dilation을 쓰면 된다는 것은 알겠지만, 실제로 언제 써야 할지 판단 기준이 필요하다.


# Color Preservation & Hair Fusion 

원본 이미지의 색상과 품질을 유지하면서 자연스러운 헤어를 생성하기 위해 적용된 핵심 기술들을 정리하고, 최근 결과물을 비교했습니다.

## 1. Soft-Blending (배경 보존)
"Soft-Blending" 기법을 통해 원본 배경이 변질되지 않고 온전히 유지되었습니다.
- **적용 결과**: 생성된 헤어와 배경 사이의 합성이 부드럽게 이루어지면서도, 배경 영역의 픽셀 데이터는 변경 없이 **완벽하게 보존(Preserved)**되었습니다.
- **기대**: 단순히 잘라 붙인 느낌이 아니라, 자연스럽게 녹아들면서도 원본의 배경 정보(조명, 색감 등)를 해치지 않는 것을 기대

## 2. 정밀한 마스킹 (SegFace)
자연스러운 합성을 위해서는 마스크가 칼같이 정확해야 합니다.
- **SegFace 도입**: 기존 모델보다 월등히 정밀한 SegFace를 도입하여 마스크 품질을 높였습니다.
- **역할**: 머리카락이 심어질 위치를 픽셀 단위로 정확히 특정해줌으로써, 모델이 "어디까지가 원본이고 어디부터가 생성인지"를 명확히 구분하게 돕습니다. 이를 통해 원본 얼굴 영역을 침범하지 않고 필요한 부분에만 헤어를 생성합니다.

## 3. 모델 본연의 Blending 능력 활용 (TinyAdapterNative)
인위적인 후처리 대신, 모델 학습 단계에서 해결책을 찾았습니다.
- **Native Adapter**: 학습 시 1024px 고해상도에서 마스크 정보를 주입받아, 모델 스스로가 "경계면을 자연스럽게 잇는 법"을 학습했습니다.
- **결과**: 별도의 후처리 블러 없이도, 모델 생성 결과물 자체가 원본 톤과 자연스럽게 어우러지도록 유도됩니다.

## 4. Result Comparison (0125 vs 0128)
초기 결과(0125)와 개선된 Soft-Blending 전략이 적용된 최근 결과(0128)를 원본과 비교한 표입니다.

| ID | Original | Result (01/25) | Result (01/28) |
| :---: | :---: | :---: | :---: |
| **test1** | ![Org](test_data/original_images/test1.png) | ![0125](results/0125/test1.png) | ![0128](results/0128/test1.png) |
| **test2** | ![Org](test_data/original_images/test2.png) | ![0125](results/0125/test2.png) | ![0128](results/0128/test2.png) |
| **test3** | ![Org](test_data/original_images/test3.png) | ![0125](results/0125/test3.png) | ![0128](results/0128/test3.png) |
| **test6** | ![Org](test_data/original_images/test6.png) | ![0125](results/0125/test6.png) | ![0128](results/0128/test6.png) |

얼굴 색 진해지는 부분은 해결함. 헤어 부자연스러움은 여전히 남아있음. 자연스러움은 soft-blending 적용 전이 더 좋은 듯
이 둘 사이의 trade-off를 해결할 수 있는 방안을 고안해보아야 함.




# Siamesefc-pytorch

Object Tracking - SiameseFC Pytorch Implementation
## 필요 데이터 및 다운로드 링크
### Installation
git clone 후 requirements.txt에 있는 패키지를 설치합니다. 

### Train Dataset
1. ILSVRC2015를 다운받습니다.
2. Offline으로 Dataset을 Curation 합니다. [ILSVRC2015_VID_CURATION]

```sh
python create_curation.py --data-dir "./datasets/ILSVRC2015" --output-dir "./datasets/ILSVRC2015_VID_CURATION"

```
```sh
├─datasets
│  ├─ILSVRC2015_VID_CURATION
│  │      ILSVRC2015_train_00000001
│  │      ILSVRC2015_train_00092007 ...
│  └─imdb_video_train.json
```
### Test Dataset
1. OTB13을 다운 받습니다.
```sh
├─otb13
│  │      Basketball
│  │      Bolt
│  │      Boy
│  │      Car4 ...
```

## 학습 방법
Train Dataset과 Test Dataset을 다운로드 후 train.py를 실행시키면, 1 에폭마다 weights 폴더에 가중치를 저장합니다. 
```
python train.py --epochs=100 --batch_size=32 --lr=1e-2
```
- epochs : 에폭 (default : 100)
- batch_size : 배치사이즈 (default : 32)
- lr : learning rate (default : 1e-2)
- momentum : SGD의 모멘텀 (default : 0.9)
- weight_decay : SGD의 weight decay (default : 5e-4)
- gamma : SGD gamma (default : 0.8685)
- step_size : SGD Scheduler step size (default : 1)
- num_workers

## 테스트 방법
- OPE를 계산 후 그래프 이미지를 현재 디렉토리에 저장합니다.
- paper 폴더에 논문 저자가 만든 Tracking 결과가 있습니다.
- test.py는 기존 논문 결과는 파란색, 재현 결과는 초록색, ground truth는 빨간색으로 바운딩 박스를 그려 result 폴더에 저장합니다. 
```
python test.py --trained_model="./weights/embed_clr_70.pth"
python test.py --start_idx=5 --viz_graph="TRE" --trained_model="./weights/embed_clr_70.pth"
python test.py --scale_ratio=0.8 --viz_graph="SRE_0.8" --trained_model="./weights/embed_clr_70.pth"
python test.py --scale_ratio=0.9 --viz_graph="SRE_0.9" --trained_model="./weights/embed_clr_70.pth"
python test.py --scale_ratio=1.1 --viz_graph="SRE_1.1" --trained_model="./weights/embed_clr_70.pth"
python test.py --scale_ratio=1.2 --viz_graph="SRE_1.2" --trained_model="./weights/embed_clr_70.pth"
```
- trained_model : 학습한 모델의 가중치 경로
- start_idx : ground truth의 시작 지점, default = 0
- scale_ratio : ground truth box의 크기, default = 1.0

## GIF Generation
- test.py 실행 후 result 폴더에 있는 이미지를 gif로 만들어줍니다.
```
python generate_gif.py
```
- gif 폴더에 otb13 비디오별로 Tracking 결과가 저장됩니다. 


## 재현 결과
reimplementation.pth 가중치는 아래와 같은 성능을 냅니다. 
- OPE : 0.612 (5 scale) / 논문 0.612

![](https://github.com/lsh3163/siamesefc-pytorch/blob/main/OPE.png)

- FPS : 0.56 (5 scale) / 논문 58 fps
- Tracking example

![](https://github.com/lsh3163/siamesefc-pytorch/blob/main/tracking_example.jpg)

## Hyper-Parameter Detail
- tracker.py에서 하이퍼 파라미터를 수정할 수 있습니다. 

```python
self.scales = np.array([1.0255**(-2), 1.0255**(-1), 1.0255**(0), 1.0255**(1), 1.0255**(2)])
self.dampling = 0.35
self.cosine_weight = 0.3
self.penalty = np.ones((5)) * 0.962
self.penalty[5//2] = 1
```
- scales : 5 scale로 이미지를 축소 및 확대 합니다. (default : 5)
- dampling : 이동 반영 비율을 나타냅니다. (default : 0.35)
- cosine_weight : consine window 와 score map를 합할 때의 비율을 뜻합니다. (default : 0.3)
- penalty : 확대 및 축소에 따른 penalty 계수입니다. (default : 0.962)

# 학습 방법
1) bash shell 파일인 train.sh에 필요한 parser argument를 넣어서 `train.sh`로 실행.

2) 모델의 각 epoch 체크포인트는 `checkpoints/{모델id}`에 저장된다.

3) valid set과 test set의 모델의 예측값들은 numpy파일로 `results/{모델id}`에 저장된다.

# 추론 방법
1) model의 input sequence와 길이가 일치하는 원하는 시점의 데이터를 `dataset` 폴더 안에 `pred.csv` 이름으로 위치시킨다. 

2) bash shell 파일인 `inference.sh`에 필요한 parser argument를 넣어서 `inference.sh`로 실행.

3) 모델이 예측한 값들은 `results/{모델id}` 폴더에서 `real_prediction.csv`라는 하나의 csv 파일을 확인 가능. 


```
참고)

모든 코드의 내용들은 Are Transformers Effective for Time Series Forecasting? (AAAI 2023) 논문의 https://github.com/cure-lab/LTSF-Linear를 참고했습니다.

@inproceedings{Zeng2022AreTE,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Ailing Zeng and Muxi Chen and Lei Zhang and Qiang Xu},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}

```
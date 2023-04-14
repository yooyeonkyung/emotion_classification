# emotion_classification


## 1. 코드 설명

- [KEMDy20](https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR) 데이터셋은 전체 데이터 13,462개 사용.
- 발화 세그먼트 텍스트와 중복 레이블이 포함된 레이블을 이용한 멀티 레이블 데이터 생성.
- 텍스트에 대해 태깅을 제거하는 과정 진행 후 학습에 이용.
- 학습과 평가에서 사용되는 데이터셋은 8:2로 분배하여 사용.
- 레이블이 학습과 평가 과정에 고르게 분배되도록 sklearn의 MultilabelStratifiedKFold 를 이용하여 5-fold crossvalidation dataset 생성.

#### Installed Version

```
torch: 1.10.0
python: 3.7.13
numpy: 1.19.5
pandas: 1.3.5
transformers: 4.18.0
gluonnlp: 0.10.0
scikit-learn: 1.0.2
matplotlib: 3.5.3

```

#### How to install KoBERT
```
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
```

#### Training Random Seed
```
RANDOM_SEED = 11
````

---

## 2. 코드 실행방식에 대한 설명

#### Configs default value
```
--batch_size 8
-- epochs 10
-- shuffle True
-- th False # threshold 적용 여부
```

##### (1) 랜덤 배치 순서 감정 분류 학습&평가 모델 실행

  train_data: kem20_tr0.csv / val_data: kem20_vl0.csv
  ```
  python main.py
  ```


##### (2) 성능이 가장 좋은 모델의 파라미터값을 이용한 임계값 획득

```
threshold.ipynb
```

##### (3) 임계값을 적용한 랜덤 배치 순서 적용 감정 분류 학습&평가 모델 실행

   train_data: kem20_tr0.csv / val_data: kem20_vl0.csv

  ```
  python main.py --th True
  ```


##### (4) 성능이 우수한 모델의 파라미터를 사용하여 랜덤 배치 기준 커리큘럼 스코어 코드 실행

  kemdy20: kem20_tr0.csv
  
  ```
  python scoring.py  # score_0.csv
  ```


##### (5) 커리큘럼 스코어 값을 이용한 커리큘럼 데이터 획득

  score: score_0.csv / kemdy20: kem20_tr0.csv
  ```
  curriculum_data.ipynb  # kem20_h2l_0.csv
  ```

##### (6) 임계값을 적용한 커리큘럼 순서 적용 감정 분류 학습&평가 모델 실행

  train_data: kem20_h2l_0.csv / val_data: kem20_vl0.csv
  
  ```
  python main.py --shuffle False --th True
  ```

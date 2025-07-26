# 전력 사용량 예측 프로젝트

Dacon 전력 사용량 예측 대회를 위한 머신러닝 프로젝트입니다.

## 📁 프로젝트 구조

```
Electricity-Usage-Prediction/
├── data/                    # 데이터 파일들
│   ├── train.csv           # 훈련 데이터
│   ├── test.csv            # 테스트 데이터
│   ├── building_info.csv   # 건물 정보
│   └── sample_submission.csv
├── notebooks/              # Jupyter 노트북
│   └── first_test.ipynb   # 메인 분석 노트북
├── src/                    # 소스 코드
│   └── electricity_prediction.py
├── submissions/            # 제출 파일들
│   ├── submission_weighted.csv
│   └── submission_stacked.csv
├── .venv/                  # 가상환경
├── pyproject.toml          # 프로젝트 설정
├── uv.lock                 # 의존성 잠금 파일
└── README.md
```

## 🚀 시작하기

### 1. 가상환경 활성화
```bash
# Windows
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 2. 의존성 설치
```bash
uv sync
```

### 3. 모델 실행
```bash
# Python 스크립트 실행
python src/electricity_prediction.py

# 또는 Jupyter 노트북 실행
jupyter notebook notebooks/first_test.ipynb
```

## 📊 모델 정보

### 사용된 모델
- **LightGBM**: 그래디언트 부스팅
- **XGBoost**: 그래디언트 부스팅
- **CatBoost**: 그래디언트 부스팅

### 앙상블 방법
1. **가중 앙상블**: 성능에 반비례하는 가중치 적용
2. **스태킹 앙상블**: 메타 모델을 통한 앙상블

### 피처 엔지니어링
- 시간 관련: hour, dayofweek, month, is_weekend, season
- 설비 관련: has_solar, has_ess, has_pcs
- 상호작용: 기온x태양광, 기온xESS
- 물리적: 체감온도, 불쾌지수

## 📈 결과

- **평가 지표**: SMAPE (Symmetric Mean Absolute Percentage Error)
- **제출 파일**: `submissions/` 폴더에 저장

## 🔧 개발 환경

- Python 3.13.5
- uv (패키지 관리자)
- 주요 라이브러리: pandas, numpy, scikit-learn, lightgbm, xgboost, catboost

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

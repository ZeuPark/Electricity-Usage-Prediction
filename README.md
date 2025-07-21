# Electricity-Usage-Prediction

Dacon 전력 사용량 예측 프로젝트

## 환경 설정

이 프로젝트는 [uv](https://github.com/astral-sh/uv)를 사용하여 의존성을 관리합니다.

### 사전 요구사항

- Python 3.8 이상
- uv 설치: `pip install uv`

### 설치 및 실행

1. **가상환경 생성 및 의존성 설치**
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

2. **가상환경 활성화**
   ```bash
   source .venv/bin/activate  # macOS/Linux
   # 또는
   .venv\Scripts\activate     # Windows
   ```

3. **Jupyter 커널 설치** (이미 완료됨)
   ```bash
   python -m ipykernel install --user --name=dacon-electricity --display-name="Dacon Electricity"
   ```

4. **Jupyter 노트북 실행**
   ```bash
   jupyter notebook
   ```

## 사용된 라이브러리

- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **머신러닝**: scikit-learn, lightgbm, xgboost, catboost
- **진행률 표시**: tqdm
- **개발 환경**: jupyter, ipykernel

## 프로젝트 구조

```
dacon_electricity_usage_pred/
├── data/                    # 데이터 파일들
├── first_test.ipynb        # 메인 분석 노트북
├── requirements.txt        # Python 의존성
├── pyproject.toml         # 프로젝트 설정 (uv용)
├── .venv/                 # 가상환경 (자동 생성)
└── README.md             # 프로젝트 설명서
```

## 노트북 사용법

1. Jupyter 노트북을 실행합니다
2. 커널 선택에서 "Dacon Electricity"를 선택합니다
3. `first_test.ipynb` 파일을 열어 분석을 시작합니다

## 의존성 관리

새로운 패키지를 추가하려면:

```bash
# 가상환경 활성화 후
uv pip install <package-name>
uv pip freeze > requirements.txt
```

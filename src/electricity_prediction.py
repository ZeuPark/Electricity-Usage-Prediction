#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전력 사용량 예측 모델링
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    """SMAPE (Symmetric Mean Absolute Percentage Error) 계산"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100

def prepare_data(df):
    """데이터를 훈련/테스트로 분리하고 전처리"""
    drop_cols = ['num_date_time', '일시', 'is_train']
    target_col = '전력소비량(kWh)'
    
    # 훈련/테스트 데이터 분리
    train_df = df[df['is_train'] == 1].copy()
    test_df = df[df['is_train'] == 0].copy()
    
    # 특성과 타겟 분리
    X_train = train_df.drop(columns=drop_cols + [target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=drop_cols + [target_col])
    
    print(f"훈련 데이터 크기: {X_train.shape}")
    print(f"테스트 데이터 크기: {X_test.shape}")
    print(f"특성: {list(X_train.columns)}")
    
    return X_train, y_train, X_test

def cross_val_ensemble(X, y, X_test, folds=5):
    """교차 검증을 통한 앙상블 예측"""
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    # 예측 저장 배열
    preds_lgbm = np.zeros(X_test.shape[0])
    preds_xgb = np.zeros(X_test.shape[0])
    preds_cat = np.zeros(X_test.shape[0])
    
    # 검증 점수 저장
    scores_lgbm = []
    scores_xgb = []
    scores_cat = []

    print("교차 검증 시작...")
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), desc="Cross Validation", total=folds)):
        print(f"\nFold {fold + 1}/{folds} 처리 중...")
        
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 모델 훈련
        print("  - LightGBM 훈련 중...")
        lgbm = LGBMRegressor(random_state=42, verbose=-1)
        lgbm.fit(X_tr, y_tr)

        print("  - XGBoost 훈련 중...")
        xgb = XGBRegressor(random_state=42, verbosity=0)
        xgb.fit(X_tr, y_tr)

        print("  - CatBoost 훈련 중...")
        cat = CatBoostRegressor(random_state=42, verbose=0)
        cat.fit(X_tr, y_tr)

        # 검증 예측
        val_lgbm = lgbm.predict(X_val)
        val_xgb = xgb.predict(X_val)
        val_cat = cat.predict(X_val)

        # 검증 점수 계산
        score_lgbm = smape(y_val, val_lgbm)
        score_xgb = smape(y_val, val_xgb)
        score_cat = smape(y_val, val_cat)
        
        scores_lgbm.append(score_lgbm)
        scores_xgb.append(score_xgb)
        scores_cat.append(score_cat)
        
        print(f"  - Fold {fold + 1} 점수 - LGBM: {score_lgbm:.4f}, XGB: {score_xgb:.4f}, CAT: {score_cat:.4f}")

        # 테스트 예측 (평균화를 위해 누적)
        print("  - 테스트 데이터 예측 중...")
        preds_lgbm += lgbm.predict(X_test) / folds
        preds_xgb += xgb.predict(X_test) / folds
        preds_cat += cat.predict(X_test) / folds

    # 평균 점수 출력
    mean_lgbm = np.mean(scores_lgbm)
    mean_xgb = np.mean(scores_xgb)
    mean_cat = np.mean(scores_cat)
    
    print(f"\n=== 교차 검증 결과 ===")
    print(f"LightGBM SMAPE: {mean_lgbm:.4f} (±{np.std(scores_lgbm):.4f})")
    print(f"XGBoost SMAPE: {mean_xgb:.4f} (±{np.std(scores_xgb):.4f})")
    print(f"CatBoost SMAPE: {mean_cat:.4f} (±{np.std(scores_cat):.4f})")

    return (
        preds_lgbm, preds_xgb, preds_cat,
        mean_lgbm, mean_xgb, mean_cat
    )

def weighted_ensemble(p1, p2, p3, w1, w2, w3):
    """가중 평균 앙상블"""
    return w1 * p1 + w2 * p2 + w3 * p3

def stacking_ensemble(X, y, X_test):
    """스태킹 앙상블"""
    print("스태킹 앙상블 훈련 중...")
    
    base_models = [
        ('lgbm', LGBMRegressor(random_state=42, verbose=-1)),
        ('xgb', XGBRegressor(random_state=42, verbosity=0)),
        ('cat', CatBoostRegressor(random_state=42, verbose=0))
    ]
    
    stack_model = StackingRegressor(
        estimators=base_models, 
        final_estimator=LinearRegression(),
        cv=5
    )
    
    stack_model.fit(X, y)
    predictions = stack_model.predict(X_test)
    
    return predictions

def main():
    """메인 실행 함수"""
    print("=== 전력 사용량 예측 모델링 시작 ===")
    
    # 1. 데이터 로드
    print("데이터 로드 중...")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    building = pd.read_csv('data/building_info.csv')
    
    # 일조, 일사 열 제거
    train = train.drop(columns=['일조(hr)', '일사(MJ/m2)'])
    
    # train/test 구분
    train['is_train'] = 1
    test['is_train'] = 0
    
    # 데이터 합치기
    combined_df = pd.concat([train, test], ignore_index=True)
    
    # 2. 건물 정보 전처리
    print("건물 정보 전처리 중...")
    cols_to_convert = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
    for col in cols_to_convert:
        building[col] = pd.to_numeric(building[col].replace('-', np.nan))
        building[col] = building[col].fillna(0)
    
    # 건물 정보 병합
    merged_df = combined_df.merge(
        building[['건물번호'] + cols_to_convert],
        on='건물번호',
        how='left'
    )
    
    # 3. 피처 엔지니어링
    print("피처 엔지니어링 중...")
    
    # 날짜 변환
    merged_df['일시'] = pd.to_datetime(merged_df['일시'], format='%Y%m%d %H')
    
    # 시간 기반 피처 생성
    merged_df['hour'] = merged_df['일시'].dt.hour
    merged_df['dayofweek'] = merged_df['일시'].dt.dayofweek
    merged_df['month'] = merged_df['일시'].dt.month
    merged_df['is_weekend'] = (merged_df['dayofweek'] >= 5).astype(int)
    
    # 계절 정보
    def get_season(month):
        if month in [3, 4, 5]: return 1  # 봄
        elif month in [6, 7, 8]: return 2  # 여름
        elif month in [9, 10, 11]: return 3  # 가을
        else: return 4  # 겨울
    
    merged_df['season'] = merged_df['month'].map(get_season)
    
    # 설비 설치 여부
    merged_df['has_solar'] = (merged_df['태양광용량(kW)'] > 0).astype(int)
    merged_df['has_ess'] = (merged_df['ESS저장용량(kWh)'] > 0).astype(int)
    merged_df['has_pcs'] = (merged_df['PCS용량(kW)'] > 0).astype(int)
    
    # 상호작용 피처
    merged_df['기온x태양광'] = merged_df['기온(°C)'] * merged_df['has_solar']
    merged_df['기온xESS'] = merged_df['기온(°C)'] * merged_df['has_ess']
    
    # 체감온도
    merged_df['체감온도'] = merged_df['기온(°C)'] - ((0.55 - 0.0055 * merged_df['습도(%)']) * (merged_df['기온(°C)'] - 14.5))
    
    # 불쾌지수
    merged_df['불쾌지수'] = 0.81 * merged_df['기온(°C)'] + 0.01 * merged_df['습도(%)'] * (0.99 * merged_df['기온(°C)'] - 14.3) + 46.3
    
    # 4. 데이터 준비
    X_train, y_train, X_test = prepare_data(merged_df)
    
    # 5. 교차 검증 앙상블
    preds_lgbm, preds_xgb, preds_cat, smape_lgbm, smape_xgb, smape_cat = cross_val_ensemble(
        X_train, y_train, X_test, folds=5
    )
    
    # 6. 가중 앙상블
    scores = [smape_lgbm, smape_xgb, smape_cat]
    weights = 1 / np.array(scores)  # 점수에 반비례
    weights = weights / weights.sum()  # 정규화
    
    print(f"\n=== 가중치 ===")
    print(f"LightGBM: {weights[0]:.4f}")
    print(f"XGBoost: {weights[1]:.4f}")
    print(f"CatBoost: {weights[2]:.4f}")
    
    pred_weighted = weighted_ensemble(
        preds_lgbm, preds_xgb, preds_cat, 
        weights[0], weights[1], weights[2]
    )
    
    # 7. 스태킹 앙상블
    pred_stacked = stacking_ensemble(X_train, y_train, X_test)
    
    # 8. 제출 파일 생성
    print("\n=== 제출 파일 생성 ===")
    
    # 테스트 데이터의 num_date_time 추출
    test_ids = merged_df[merged_df['is_train'] == 0]['num_date_time'].copy()
    
    # submission 파일 생성
    submission_weighted = pd.DataFrame({
        'num_date_time': test_ids,
        'answer': pred_weighted
    })
    
    submission_stacked = pd.DataFrame({
        'num_date_time': test_ids,
        'answer': pred_stacked
    })
    
    # 파일 저장
    submission_weighted.to_csv('submission_weighted.csv', index=False)
    submission_stacked.to_csv('submission_stacked.csv', index=False)
    
    print("제출 파일 생성 완료:")
    print("- submission_weighted.csv (가중 앙상블)")
    print("- submission_stacked.csv (스태킹 앙상블)")
    
    # 9. 결과 요약
    print("\n=== 예측 결과 요약 ===")
    print(f"가중 앙상블 예측 범위: {pred_weighted.min():.2f} ~ {pred_weighted.max():.2f}")
    print(f"스태킹 앙상블 예측 범위: {pred_stacked.min():.2f} ~ {pred_stacked.max():.2f}")

if __name__ == "__main__":
    main() 
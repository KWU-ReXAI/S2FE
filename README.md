# 기본적 분석 및 머신러닝 앙상블 모델 기반 주식 종목 선택 (S2FE)

## 1. 프로젝트 개요

이 저장소는 "기본적 분석 및 머신러닝 앙상블 모델 기반의 주식 종목 선택 (S2FE)" 연구의 Python 구현 코드입니다.

본 프로젝트는 KOSPI 200 종목을 대상으로 기업의 **재무제표**, **공시정보**, 그리고 **거시경제지표** 데이터를 통합적으로 분석합니다.

S2FE 모델은 **MLP (Multi-Layer Perceptron)**, **ANFIS (Adaptive-Network-based Fuzzy Inference System)**, **RF (Random Forest)**를 결합한 앙상블 모델을 사용하여 주식의 미래 시장 초과 수익률을 예측하고, 우수한 성과를 내는 종목을 선별합니다.

## 2. 핵심 기능

* **다중 데이터 소스 통합**: 재무, 공시, 거시경제 데이터를 결합하여 기업의 내/외부적 요인을 종합적으로 고려합니다.
* **피처 엔지니어링**: 결측치 처리 (SoftImpute), 변화율 계산, 시장 초과 수익률 (Relative Return) 라벨링을 수행합니다.
* **특징 선택**: VIF (분산 팽창 인수)를 통해 다중공선성을 제거하고, Random Forest, 전진/후진 선택법(FS/BE)을 통해 핵심 특징을 선별합니다.
* **계층적 앙상블 모델**:
    * **전체 섹터 모델 (All-Sector Model)**: 모든 종목을 대상으로 학습된 범용 모델.
    * **섹터 클러스터 모델 (Sector-Cluster Model)**: 유사한 특징을 가진 섹터들을 군집화하여 학습한 특화 모델.
* **종목 선정 전략**: 전체 섹터 모델과 섹터 클러스터 모델의 예측 결과를 **교집합(Intersection)**하여 안정적이고 수익성 높은 종목을 최종 선택합니다.
* **백테스팅**: 워크 포워드(Walk-Forward) 방식을 사용하여 여러 기간(Phase)에 걸쳐 모델의 성과(CAGR, Sharpe Ratio, MDD)를 검증합니다.
* **모델 해석력 (XAI)**: SHAP 라이브러리를 사용해 앙상블 모델(MLP, ANFIS, RF)의 예측에 각 특징이 얼마나 기여했는지 분석합니다.

### 주요 파일 설명

* **`run.py`**: `datapreprocessing.py`, `train.py`, `test.py`를 순차적으로 실행하는 메인 스크립트입니다.
* **`datapreprocessing.py`**:
    * 재무, 공시, 거시경제 데이터를 로드하고 병합합니다.
    * `SoftImpute`를 사용해 결측치를 대치합니다.
    * 특징(변화율)과 라벨(시장 초과 수익률)을 계산합니다.
    * `calculate_vif_iteratively` (VIF)를 통해 다중공선성 변수를 제거합니다.
    * `random_forest_feature_selection` 등을 이용해 최종 학습 특징을 선별합니다.
* **`datamanager.py`**:
    * 워크 포워드(Walk-Forward) 방식의 학습/검증/테스트 기간(Phase)을 관리합니다.
* **`model.py`**:
    * `MultiLayerPerceptron`: PyTorch 기반의 MLP 모델 정의.
    * `AggregationModel`: MLP, RF, ANFIS를 앙상블하는 모델.
    * `MyModel`: `AggregationModel`을 사용하여 전체 섹터 모델 및 섹터 클러스터 모델을 학습하고, `backtest` 메서드를 통해 백테스팅을 수행하는 메인 클래스.
* **`train.py`**:
    * `MyModel`을 초기화하고 `trainALLSectorModels`, `trainClusterModels`를 호출하여 모델을 학습시킵니다.
    * 각 Phase별로 학습된 모델(`model.joblib`)을 `result/` 디렉토리에 저장합니다.
* **`test.py`**:
    * `train.py`에서 저장된 모델을 로드합니다.
    * `model.backtest`를 실행하여 테스트 기간의 성과(CAGR, Sharpe, MDD)를 측정하고 결과를 CSV와 그래프로 저장합니다.

## 4. 실행 방법

1.  **데이터 준비**
    * 프로젝트에서 요구하는 형식에 맞게 원본 데이터를 `data_kr/` 등의 디렉토리에 배치합니다. (예: `data_kr/price`, `data_kr/merged`, `data_kr/public_info` 등)

2.  **필요 라이브러리 설치**
    ```bash
    pip install -r requirement.txt
    ```
    - 필요에 따라 추가적인 pip install이 필요할 수도 있습니다.
3. **Dataset 다운로드**
    ```bash
    python set_data.py
    ```
    - ./data_kr/kospi200.txt에 {code,name,sector}로 이루어진 종목 정보가 사전에 존재해야 합니다.
    - ./data_kr/macro_economic/{거시경제지표명}.csv가 사전에 존재해야 merged.csv로 통합됩니다.
4. **전체 프로세스 실행**
    * `run.py` 스크립트를 실행하면 데이터 전처리, 모델 학습, 백테스팅이 순차적으로 진행됩니다.
    ```bash
    python run.py
    ```
    * 개별 스크립트(`datapreprocessing.py`, `train.py`, `test.py`)를 순서대로 직접 실행할 수도 있습니다.

5. **결과 확인**
    * 학습된 모델과 학습 성과는 `result/{모델명}/train_result_dir_{N}/`에 저장됩니다.
    * 최종 백테스팅 결과(CSV 리포트, 그래프)는 `result/{모델명}/test_result_dir/`에 저장됩니다.

## 5. 모델 방법론 (S2FE)

본 프로젝트는 논문의 워크플로우를 따릅니다.

1.  **데이터 수집**: 재무제표, 공시정보, 거시경제지표 데이터를 수집합니다.
2.  **데이터 전처리**:
    * 결측치 대치 (`SoftImpute`), 특징(변화율) 및 라벨(시장 초과 수익률) 생성.
    * 차원 축소 (VIF) 및 특징 선택 (RF)을 적용합니다.
    * 데이터를 '전체 섹터'와 '섹터 클러스터'로 분리하여 처리합니다.
3.  **예측 모델링**:
    * '전체 섹터 모델'과 '섹터 클러스터 모델' 각각에 대해 앙상블 모델(MLP + ANFIS + RF)을 학습시킵니다.
4.  **종목 선택 (교집합)**:
    * 두 모델이 예측한 수익률을 기준으로 각각 상위 K%의 종목을 선택합니다.
    * 두 그룹의 **교집합**에 해당하는 종목들을 최종 포트폴리오로 선정합니다.
5.  **백테스팅**:
    * 워크 포워드 방식을 사용해 정해진 기간(Phase)별로 리밸런싱을 수행하며 누적 수익률(CAGR), 위험 대비 수익(Sharpe), 최대 손실(MDD)을 측정합니다.
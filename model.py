import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import joblib
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from Utils import Utils
from torch.nn import Linear, ReLU, GRU
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from datetime import datetime, timedelta
from functools import reduce

from membership import make_anfis
from experimental import train_anfis
from datamanager import DataManager

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(MultiLayerPerceptron,self).__init__()

        self.network = nn.Sequential(
            Linear(input_size, hidden_size, device=device),
            ReLU(),
            Linear(hidden_size, hidden_size, device=device),
            ReLU(),
            Linear(hidden_size, hidden_size, device=device),
            ReLU(),
            Linear(hidden_size, 1, device=device)
        )

        self.importance= {}
        self.optimal_weights = {}

    def forward(self,x):
        z = self.network(x)
        return z

    def fit(self, data, lr, epochs):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            for x, y in data:
                optimizer.zero_grad()
                pred = self(x)
                loss = criterion(pred.squeeze(), y)
                loss.backward()
                optimizer.step()

class AggregationModel:
    def __init__(self, n_input, n_rules, hidden_layer, device, aggregate = ""):
        self.mlp = MultiLayerPerceptron(n_input, hidden_layer, device)
        self.rf = RandomForestRegressor()
        self.device = device

    def fit(self, X, y, epochs_mlp, epochs_anfis, lr_mlp, lr_anfis):
        data = DataLoader(TensorDataset(X, y), batch_size = 64, shuffle = True)
        self.mlp.fit(data, lr_mlp, epochs_mlp)
        self.anfis = make_anfis(X, 2, 1)
        train_anfis(self.anfis, data, epochs_anfis)
        self.rf = self.rf.fit(X.cpu().numpy(), y.cpu().numpy())

    def predict(self, X, symbol_index, agg2 = False):
        if agg2 != True: pred_anfis = self.anfis(X).cpu().detach().numpy().squeeze()
        pred_mlp = self.mlp(X).cpu().detach().numpy().squeeze()
        pred_rf = self.rf.predict(X.cpu().numpy()).squeeze()

        if agg2 != True:
            sum_prediction = pred_rf + pred_mlp + pred_anfis
        else:
            sum_prediction = pred_rf + pred_mlp
        sorted_rank = pd.Series(sum_prediction.squeeze(), index=symbol_index).sort_values(ascending=False)

        return sorted_rank

    def loss(self, X, y, agg2 = False):
        if agg2 != True: pred_anfis = self.anfis(X).cpu().detach().numpy().squeeze()
        pred_mlp = self.mlp(X).cpu().detach().numpy().squeeze()
        pred_rf = self.rf.predict(X.cpu().numpy()).squeeze()

        if agg2 != True:
            mean_prediction = (pred_rf + pred_mlp + pred_anfis)/3
        else:
            mean_prediction = (pred_rf + pred_mlp)/2

        mean_prediction = torch.Tensor(mean_prediction)
        criterion = nn.MSELoss()
        loss = criterion(mean_prediction, y.cpu())

        return loss.item()

class MyModel(nn.Module):
    def __init__(self, feature_n, valid_stock_k, valid_sector_k, each_sector_stock_k, final_stock_k, phase, device,
                 ensemble="S3CE", clustering=False, cluster_n=5, epochs_MLP = 300, epochs_anfis = 100, lr_MLP = 0.0001, lr_anfis = 0.01, hidden = 128
                 ,isLLMexperiment = False, sector_name = []):
        # 클래스 초기화
        super(MyModel, self).__init__()
        self.feature_n = feature_n  # 사용할 재무 feature 개수
        self.valid_stock_k = valid_stock_k  # 검증 데이터에서 선택할 주식 수
        self.valid_sector_k = valid_sector_k  # 검증 데이터에서 선택할 섹터 수
        self.each_sector_stock_k = each_sector_stock_k  # 각 섹터에서 선택할 주식 수
        self.final_stock_k = final_stock_k  # 최종적으로 선택할 주식 수
        self.phase = phase  # 특정 실험의 데이터 기간
        self.ensemble = ensemble  # 사용할 앙상블 기법(MLP, RF, Aggregation 등)
        self.clustering = clustering  # 섹터 군집화 활성화 여부

        self.epochs_MLP = epochs_MLP  # MLP 학습 epoch
        self.lr_MLP = lr_MLP  # 0.001
        self.hidden = hidden  # MLP 은닉층 크기

        self.epochs_anfis = epochs_anfis
        self.lr_anfis = lr_anfis
        self.n_rules = 10

        self.device = device

        self.DM = DataManager(features_n=feature_n, cluster_n=cluster_n)  # 데이터 관리 클래스
        self.DM.create_date_list()
        self.Utils = Utils()  # 수익률 계산 등 유틸리티 함수 제공

        self.valid_models = {}
        self.sector_models = {}
        if isLLMexperiment: self.cluster_list = self.DM.get_clusters_of_sectors(sector_name)
        else: self.cluster_list = self.DM.cluster_list

    def recordParameter(self):
        file_path = "./result/train_parameter.csv"
        new_data = [
            {"Parameter": "feature_n", "Value": self.feature_n},
            {"Parameter": "valid_stock_k", "Value": self.valid_stock_k},
            {"Parameter": "valid_sector_k", "Value": self.valid_sector_k},
            {"Parameter": "each_sector_stock_k", "Value": self.each_sector_stock_k},
            {"Parameter": "final_stock_k", "Value": self.final_stock_k},
            {"Parameter": "ensemble", "Value": self.ensemble},
            {"Parameter": "clustering", "Value": self.clustering},
            {"Parameter": "epochs_MLP", "Value": self.epochs_MLP},
            {"Parameter": "lr_MLP", "Value": self.lr_MLP},
            {"Parameter": "hidden", "Value": self.hidden},
            {"Parameter": "epochs_anfis", "Value": self.epochs_anfis},
            {"Parameter": "lr_anfis", "Value": self.lr_anfis},
            {"Parameter": "n_rules", "Value": self.n_rules}
        ]
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Parameter", "Value"])
        df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)
        df.to_csv(file_path, index=False)

    def trainClusterModels(self, withValidation=False, withLLM=False):
        print(f"trainClusterModels ({self.phase}), with validation {withValidation}, with LLM {withLLM}:")
        train_start = self.DM.phase_list[self.phase][0]
        valid_start = self.DM.phase_list[self.phase][1]
        test_start = self.DM.phase_list[self.phase][2]
        test_end = self.DM.phase_list[self.phase][3]

        print(
            f"train: {self.DM.pno2date(train_start)} ~ {self.DM.pno2date(valid_start - 1)} / valid: {self.DM.pno2date(valid_start)} ~ {self.DM.pno2date(test_start - 1)}"
            f" / test: {self.DM.pno2date(test_start)} ~ {self.DM.pno2date(test_end - 1)}")
        for sector in self.cluster_list:
            train_data, valid_data, _ = self.DM.data_phase(sector, self.phase, cluster=self.clustering, LLM = withLLM)
            if withValidation: train_data = np.concatenate((train_data, valid_data), axis=0)
            a, b = train_data.shape[0], train_data.shape[1]
            train_data = train_data.reshape(a * b, -1)
            train_data = torch.Tensor(train_data, ).to(self.device)
            the_model = AggregationModel(train_data.shape[1] - 1, self.n_rules, self.hidden, self.device)
            the_model.fit(train_data[:, :-1], train_data[:, -1], self.epochs_MLP, self.epochs_anfis, self.lr_MLP,
                          self.lr_anfis)

            self.sector_models[sector] = the_model

    def trainALLSectorModels(self, withValidation = False): # 전체 섹터를 하나의 모델로 학습
        # 전체 섹터 학습 모델
        print(f"trainALLSectorModels ({self.phase}), with validation {withValidation}: ")
        train_start = self.DM.phase_list[self.phase][0]
        valid_start = self.DM.phase_list[self.phase][1]
        test_start = self.DM.phase_list[self.phase][2]
        test_end = self.DM.phase_list[self.phase][3]

        print(
            f"train: {self.DM.pno2date(train_start)} ~ {self.DM.pno2date(valid_start - 1)} / valid: {self.DM.pno2date(valid_start)} ~ {self.DM.pno2date(test_start - 1)}"
            f" / test: {self.DM.pno2date(test_start)} ~ {self.DM.pno2date(test_end - 1)}")
        train_data = np.ndarray([0])
        train_tmp, valid_tmp, _ = self.DM.data_phase("ALL",self.phase)
        if withValidation: train_tmp = np.concatenate((train_tmp, valid_tmp))
        train_data = train_tmp.reshape(train_tmp.shape[0]*train_tmp.shape[1],-1)
        # 전체 데이터를 가져와 train_data로 변환

        self.all_sector_model = AggregationModel(train_data.shape[1] - 1, self.n_rules, self.hidden, self.device)
        train_data = torch.Tensor(train_data).to(self.device)
        self.all_sector_model.fit(train_data[:, :-1], train_data[:, -1], self.epochs_MLP, 200, self.lr_MLP,
                                  self.lr_anfis)
        # 개별 섹터 모델과 비교하기 위해 전체 시장을 학습한 모델을 실험

    def copymodels(self):
        for sector in self.DM.sector_list:
            self.sector_models[sector] = deepcopy(self.all_sector_model)

    def save_models(self,dir):
        joblib.dump(self,f"{dir}/model.joblib")

    def get_rows_by_date_range(self, code: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
        code = str(code).zfill(6)
        file_path = f"./result/experiment/{code}.csv"
        # CSV 읽기
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        # upload_dt를 datetime으로 변환
        df['upload_dt'] = pd.to_datetime(df['upload_dt'])

        # 문자열을 datetime으로 변환
        start_dt = pd.to_datetime(start_date_str)
        end_dt = pd.to_datetime(end_date_str)-timedelta(days=7)

        # 두 구간이 겹치는 조건:
        # interval.after <= end_dt  AND  interval.before >= start_dt
        mask = (start_dt <= df['upload_dt']) & (df['upload_dt'] <= end_dt)

        # 겹치는 행 반환
        return df.loc[mask].reset_index(drop=True)

    def LLM_task2(self, model_list, start_date, end_date):
        model_list = model_list.index.to_list()
        df_list = []

        start_datetime = self.DM.get_disclosure_date(start_date)
        end_datetime = self.DM.get_disclosure_date(end_date)

        for code in model_list:
            print(f"{code} of {start_date}~{end_date} : {start_datetime}~{end_datetime}")
            df = self.get_rows_by_date_range(code, start_datetime, end_datetime)
            if len(df) == 0: continue
            df_list.append(df)  # 리스트에 담기

        df_select = []
        for i in range(len(df_list)):
            df_now = df_list[i]
            for j in range(0, len(df_now), 4):
                chunk = df_now.iloc[j:j + 4].copy()  # 원본 건드리지 않기 위해 복사

                # score 컬럼이 없다면 0으로 초기화
                if "score" not in chunk.columns:
                    chunk["score"] = 0

                # 행 인덱스에서 1을 뺀 값을 더함
                chunk.reset_index(drop=True, inplace=True)
                chunk["score"] += (chunk.index)

                if "code" in chunk.columns:
                    score_sum = chunk["score"].sum()
                    code_val = chunk["code"].iloc[0]
                    df_select.append(pd.DataFrame({"code": [code_val], "score": [score_sum]}))
                else:
                    print(f"[경고] chunk에 'code' 열이 없습니다. index: {j}")

        return model_list

    def LLM_task(self, model_list, start_date, end_date):
        balance = 100000000 # 1억원
        
        # 1) model_list를 index 리스트로 변환
        if model_list is None: return None

        model_list = model_list.index.to_list()

        # 2) 조회 기간의 disclosure date 얻기
        start_datetime = self.DM.get_disclosure_date(start_date)
        end_datetime = self.DM.get_disclosure_date(end_date)

        # 3) 개별 코드별로 DataFrame 모아두기
        df_list = []
        for code in model_list:
            print(f"{code} of {start_date}~{end_date} : {start_datetime}~{end_datetime}")
            df = self.get_rows_by_date_range(code, start_datetime, end_datetime)
            if df.empty:
                continue
            df_list.append(df)

        # 4) month, code, score, start_date, end_date를 저장할 빈 DataFrame 생성
        df_select = pd.DataFrame(columns=[
            "month", "code", "score", "start_date", "end_date"
        ])

        for df_now in df_list:            
            for j in range(0, len(df_now), 4):
                chunk = df_now.iloc[j:j + 4].copy()

                chunk.reset_index(drop=True, inplace=True)
                chunk["score"] += chunk.index #가중치 넣음! 우선 임의로

                # score 합계 및 code 추출
                score_sum = chunk["score"].sum()
                code_val = chunk["code"].iloc[0]

                # filtering에서 날짜 파싱
                first_filt = chunk["filtering"].iloc[0]  # ex) "after:2021-03-22 before:2021-03-31"
                last_filt = chunk["filtering"].iloc[-1]
                start_dt = first_filt.split("after:")[1].split()[0]
                end_dt = last_filt.split("before:")[1].split()[0]

                # 한 행으로 추가
                df_select.loc[len(df_select)] = [
                    j / 4, code_val, score_sum, start_dt, end_dt
                ]

        # 6) threshold 설정 및 month별로 필터링하여 df_result 생성
        threshold = 3
        df_result = {}
        for month, grp in df_select.groupby("month"):
            filtered = (
                grp.loc[grp["score"] >= threshold,
                ["code", "score", "start_date", "end_date"]]
                .reset_index(drop=True)
            )
            df_result[int(month)] = filtered

        # 7) 결과 반환 (딕셔너리 형태)
        return df_result
    
    def LLM_task3(self, model_list, start_date, end_date):
        balance = 100000000 # 1억
        
        # 1) model_list를 index 리스트로 변환
        if model_list is None: return None

        model_list = model_list.index.to_list()

        # 2) 조회 기간의 disclosure date 얻기
        start_datetime = self.DM.get_disclosure_date(start_date)
        end_datetime = self.DM.get_disclosure_date(end_date)           
        
        # 3) 개별 코드별로 DataFrame 모아두기
        df_list = []
        for code in model_list:
            print(f"{code} of {start_date}~{end_date} : {start_datetime}~{end_datetime}")
            df = self.get_rows_by_date_range(code, start_datetime, end_datetime)
            if df.empty:
                continue
            df_list.append(df)


        start_dt = datetime.strptime(start_datetime, '%Y-%m-%d')
        end_dt = datetime.strptime(end_datetime, '%Y-%m-%d')
        current = start_dt

        while current <= end_dt:
            df_select = pd.DataFrame(columns=[
                "code", "score"
            ])
            for df_now in df_list:
                df_now['upload_dt'] = pd.to_datetime(df_now['upload_dt'])

                month_start = current
                month_end = current + relativedelta(months=1)
                chunk = df_now.loc[(month_start <= df_now['upload_dt']) & (df_now['upload_dt'] <= month_end)].copy()

                chunk.reset_index(drop=True, inplace=True)
                chunk["score"] += chunk.index #가중치 넣음! 우선 임의로

                # score 합계 및 code 추출
                score_sum = chunk["score"].sum()
                code_val = df_now["code"].iloc[0]

                # 한 행으로 추가
                df_select.loc[len(df_select)] = [
                    code_val, score_sum
                ]
            threshold = 3
            df_select = df_select[df_select['score'] >= threshold]
            
            ###### 구매 코드 구현하기 ######
            
            current += relativedelta(months=1)

        return 

    def backtest(self, verbose=True, use_all='Sector', agg='inter', inter_n=0.1, withValidation = False, isTest=True, testNum=0, dir="", withLLM = False, LLMagg = "False"):  # 백테스팅 수행
        # 선택된 섹터 및 전체 섹터 모델을 활용해 종목을 선택하고, 실제 데이터로 수익률을 평가
        # 과거 데이터를 사용하여 모델의 예측이 실제 시장에서 얼마나 잘 맞았는지를 검증하는 과정
        test_start = self.DM.phase_list[self.phase][2 if withValidation else 1]
        test_end = self.DM.phase_list[self.phase][3 if withValidation else 2]
        # 테스트 시작과 종료 시점 설정
        test_data = {}  # 섹터별 테스트 데이터를 저장할 딕셔너리
        symbols = {}  # 각 섹터별 종목(Symbol) 정보를 저장할 딕셔너리

        idx = 0
        clustered_stocks_list = []

        if use_all == "SectorAll" or use_all == "All":  # 전체 데이터를 불러옴
            _, _, all_data = self.DM.data_phase("ALL", self.phase)
            all_symbol = pd.read_csv(f"./data_kr/symbol.csv")  # 전체 섹터 데이터 가져옴

        for sector in self.cluster_list:  # 저장된 상위 섹터별 데이터를 로드
            _, _, data_tmp = self.DM.data_phase(sector, self.phase, cluster=self.clustering, LLM = withLLM)
            test_data[sector] = data_tmp

            symbol_index = pd.read_csv(f"./preprocessed_data/{sector}/symbol_index.csv")  # 해당 섹터의 주식 종목 리스트 가져옴
            symbols[sector] = symbol_index["Code"]

        ## 백테스팅 진행
        pf_mem = []  # 각 날짜별 포트폴리오 수익률 기록
        pf_mem_ks = []
        num_of_stock = []  # 매일 선택된 주식 개수 저장

        pf_mem_first = []
        pf_mem_first_ks = []

        if verbose: print(f"\n------[{self.phase}]------",flush=True)

        print(f"Test Period: {self.DM.pno2date(test_start)} ~ {self.DM.pno2date(test_end-1)}")
        for pno in range(test_start, test_end):  # test_start ~ end 기간동안 매일 반복 실행
            print(f"Test in {self.DM.pno2date(pno)}:")
            # 각 날짜마다 주식을 선택하고 수익률을 계산
            i = pno - test_start
            strdate = self.DM.pno2date(pno)  # 현재 날짜와
            next_strdate = self.DM.pno2date(pno + 1)  # 다음 날짜 반환

            stocks = pd.Series()  # 모든 섹터에서 선택된 주식을 저장
            real_last_topK_stock = []  # 최종적으로 선택된 상위 주식 저장

            for sector in self.cluster_list:  # 선택된 섹터별로 종목을 추천

                model = self.sector_models[sector]
                topK = model.predict(torch.Tensor(test_data[sector]).to(self.device)[i, :, :-1],
                                     symbols[sector])
                cnt = len(topK)
                topK = topK[:int(cnt * inter_n)]  # 상위 10% 종목 선택
                """if LLMagg == "inter":
                    # 교집합: topK와 up_code 모두에 있는 종목만
                    selected = topK[topK.index.isin(up_code)].index.to_list()
                elif LLMagg == "union":
                    # 합집합: topK와 up_code의 모든 종목
                    selected = list(set(topK.index.to_list()) | set(up_code))
                else:
                    # 기본: topK만
                    selected = topK.index.to_list()"""
                up_code = self.LLM_task3(topK, strdate, next_strdate)
                selected = topK[topK.index.isin(up_code)].index.to_list()

                real_last_topK_stock.extend(selected)

            clustered_stocks_list.append([f"{idx}"] + real_last_topK_stock)
            idx += 1
            self.final_stock_k = len(real_last_topK_stock)  # 최종적으로 선택된 주식 개수를 저장

            if verbose: print(real_last_topK_stock,flush=True)  # 선택된 최종 종목을 출력
            if isTest:
                pd.DataFrame(clustered_stocks_list).to_csv(f"./result/{dir}/test_selected_stocks_{self.phase}_{testNum}.csv", index=False)
            if not isTest:
                pd.DataFrame(clustered_stocks_list).to_csv(
                    f"{dir}/train_selected_stocks_{self.phase}.csv", index=False)
            num_of_stock.append(len(real_last_topK_stock))

            daily_change = self.Utils.get_portfolio_memory(real_last_topK_stock, strdate, next_strdate,False)
            daily_change_KOSPI = self.Utils.get_portfolio_memory(real_last_topK_stock,strdate,next_strdate,True)
            # 매일의 포트폴리오 수익률 계산
            # 선택된 주식 리스트, 현재 및 다음 날짜의 주가 데이터 이용
            pf_mem.extend(daily_change)
            pf_mem_ks.extend(daily_change_KOSPI)

        return_ratio = np.prod(np.array(pf_mem) + 1) - 1
        mdd = self.Utils.get_MDD(np.array(pf_mem) + 1)
        sharpe = self.Utils.get_sharpe_ratio(pf_mem)

        return_ratio_ks = np.prod(np.array(pf_mem_ks) + 1) - 1
        mdd_ks = self.Utils.get_MDD(np.array(pf_mem_ks) + 1)
        sharpe_ks = self.Utils.get_sharpe_ratio(pf_mem_ks)



        if verbose:
            print(f"\nMDD: {mdd}\tSharpe: {sharpe}\tCAGR: {return_ratio}",flush=True)
            print(f"KOSPI MDD: {mdd_ks}\tKOSPI Sharpe: {sharpe_ks}\tKOSPI CAGR: {return_ratio_ks}",flush=True)
            print("----------------------",flush=True)

        return return_ratio, sharpe, mdd,num_of_stock, return_ratio_ks, sharpe_ks, mdd_ks

    def save(obj):
        return (obj.__class__, obj.__dict__)

    def load(cls, attributes):
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

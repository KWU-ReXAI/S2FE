import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import joblib

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


class AggregationModel:
    def __init__(self, n_input, n_rules, hidden_layer, device, aggregate = ""):
        self.rf = RandomForestRegressor(
        n_estimators=200,    # 200개의 트리 사용 [cite: 1034]
        max_depth=10,        # 최대 깊이를 10으로 제한 [cite: 1049]
        #random_state=42,     # 결과 재현을 위한 시드 고정
        n_jobs=-1            # 모든 CPU 코어 사용
    )
        self.device = device

    def fit(self, X, y):
        self.rf = self.rf.fit(X.cpu().numpy(), y.cpu().numpy())

    def predict(self, X, symbol_index, agg2 = False):

        pred_rf = self.rf.predict(X.cpu().numpy()).squeeze()
        sorted_rank = pd.Series(pred_rf.squeeze(), index=symbol_index).sort_values(ascending=False)
        return sorted_rank

class RF_Model(nn.Module):
    def __init__(self, feature_n, valid_stock_k, valid_sector_k, each_sector_stock_k, final_stock_k, phase, device,
                 ensemble="S3CE", clustering=False, cluster_n=5, epochs_MLP = 300, epochs_anfis = 100, lr_MLP = 0.0001, lr_anfis = 0.01, hidden = 128):
        # 클래스 초기화
        super(RF_Model, self).__init__()
        self.feature_n = feature_n  # 사용할 재무 feature 개수
        self.valid_stock_k = valid_stock_k  # 검증 데이터에서 선택할 주식 수
        self.valid_sector_k = valid_sector_k  # 검증 데이터에서 선택할 섹터 수
        self.each_sector_stock_k = each_sector_stock_k  # 각 섹터에서 선택할 주식 수
        self.final_stock_k = final_stock_k  # 최종적으로 선택할 주식 수
        self.phase = phase  # 특정 실험의 데이터 기간
        self.ensemble = ensemble  # 사용할 앙상블 기법(MLP, RF, Aggregation 등)

        self.device = device

        self.DM = DataManager(features_n=feature_n, cluster_n=cluster_n)  # 데이터 관리 클래스
        self.DM.create_date_list()
        self.Utils = Utils()  # 수익률 계산 등 유틸리티 함수 제공

        self.valid_models = {}
        self.sector_models = {}
        self.cluster_list = self.DM.cluster_list

    def recordParameter(self,path):
        file_path = path
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

    def trainALLSectorModels(self, withValidation=False, model="S3CE"):  # 전체 섹터를 하나의 모델로 학습
        # 전체 섹터 학습 모델
        print(f"trainALLSectorModels ({self.phase}), with validation {withValidation}, Model {model}:")
        train_start = self.DM.phase_list[self.phase][0]
        valid_start = self.DM.phase_list[self.phase][1]
        test_start = self.DM.phase_list[self.phase][2]
        test_end = self.DM.phase_list[self.phase][3]

        print(
            f"train: {self.DM.pno2date(train_start)} ~ {self.DM.pno2date(valid_start - 1)} / valid: {self.DM.pno2date(valid_start)} ~ {self.DM.pno2date(test_start - 1)}"
            f" / test: {self.DM.pno2date(test_start)} ~ {self.DM.pno2date(test_end - 1)}")

        train_tmp, valid_tmp, _ = self.DM.data_phase("ALL", self.phase, model=model)
        if withValidation: train_tmp = np.concatenate((train_tmp, valid_tmp))
        train_data = train_tmp.reshape(train_tmp.shape[0] * train_tmp.shape[1], -1)

        self.all_sector_model = RandomForestRegressor()
        self.all_sector_model.fit(train_data[:, :-1], train_data[:, -1])



    def copymodels(self):
        for sector in self.DM.sector_list:
            self.sector_models[sector] = deepcopy(self.all_sector_model)

    def save_models(self,dir):
        joblib.dump(self,f"{dir}/model.joblib")

    def backtest(self, verbose=True, use_all='SectorAll', agg='inter', inter_n=0.1,withValidation = False, isTest=True, testNum=0, dir="", model="RF"):  # 백테스팅 수행
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
            _, _, all_data = self.DM.data_phase("ALL", self.phase,model=model)
            all_symbol = pd.read_csv(f"./data_kr/symbol.csv")  # 전체 섹터 데이터 가져옴

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

            top_all = pd.Series(self.all_sector_model.predict(all_data[i, :, :-1]),
                                index=all_symbol["code"]).sort_values(ascending=False)

            # 2. ## 수정된 부분 ##: 예측값이 0보다 큰 종목만 선택 (Long 포지션)
            long_position_stocks = top_all[top_all > 0]

            # 3. 선택된 종목의 코드(index)를 최종 포트폴리오 리스트에 추가
            real_last_topK_stock.extend(long_position_stocks.index.to_list())
            #real_last_topK_stock.extend(top_all.index[:10].to_list())

            clustered_stocks_list.append([f"{idx}"] + real_last_topK_stock)
            idx += 1

            if verbose: print(real_last_topK_stock,flush=True)  # 선택된 최종 종목을 출력
            if isTest:
                pd.DataFrame(clustered_stocks_list).to_csv(f"{dir}/test_selected_stocks_{self.phase}_{testNum}.csv", index=False)
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

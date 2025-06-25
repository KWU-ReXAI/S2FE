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

class BaggingModel:
    def __init__(self, n_modules, input_size, hidden_size):
        self.n_modules = n_modules
        self.models = []

        for i in range(n_modules):
            self.models.append(MultiLayerPerceptron(input_size, hidden_size, 'cpu'))

    def fit(self, X:torch.Tensor, y, epochs = 200, lr = 0.015):
        num_samples = int(X.shape[0] * 0.8)
        for i in range(self.n_modules):
            index = torch.randperm(X.shape[0])[:num_samples]
            sample_X = X[index]
            sample_y = y[index]
            self.models[i].fit(sample_X,sample_y,lr,epochs)

    def predict(self, X):
        outputs = [model(X).unsqueeze(0) for model in self.models]
        result = torch.cat(outputs, dim=0).mean(dim=0)
        return result.cpu().detach().numpy().squeeze()

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
                 ensemble="S3CE", clustering=False, cluster_n=5, epochs_MLP = 300, epochs_anfis = 100, lr_MLP = 0.0001, lr_anfis = 0.01, hidden = 128):
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
        self.cluster_list = self.DM.cluster_list

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

    def trainClusterModels(self, withValidation=False):
        print(f"trainClusterModels ({self.phase}), with validation {withValidation}, Model {self.ensemble}")
        train_start = self.DM.phase_list[self.phase][0]
        valid_start = self.DM.phase_list[self.phase][1]
        test_start = self.DM.phase_list[self.phase][2]
        test_end = self.DM.phase_list[self.phase][3]

        print(
            f"train: {self.DM.pno2date(train_start)} ~ {self.DM.pno2date(valid_start - 1)} / valid: {self.DM.pno2date(valid_start)} ~ {self.DM.pno2date(test_start - 1)}"
            f" / test: {self.DM.pno2date(test_start)} ~ {self.DM.pno2date(test_end - 1)}")

        if self.ensemble == "RF":
            for sector in self.cluster_list:
                train_data, valid_data, _ = self.DM.data_phase(sector, self.phase, cluster=self.clustering)
                if withValidation: train_data = np.concatenate((train_data, valid_data), axis=0)
                a,b = train_data.shape[0], train_data.shape[1]
                train_data = train_data.reshape(a*b,-1)
                the_model = RandomForestRegressor()
                the_model.fit(train_data[:,:-1], train_data[:,-1])
                self.sector_models[sector] = the_model

        elif self.ensemble == "bagging":
            for sector in self.cluster_list:
                train_data, valid_data, _ = self.DM.data_phase(sector, self.phase, cluster=self.clustering)
                if withValidation: train_data = np.concatenate((train_data, valid_data), axis=0)
                a,b = train_data.shape[0], train_data.shape[1]
                train_data = train_data.reshape(a*b,-1)
                train_data = torch.Tensor(train_data).to(self.device)
                the_model = BaggingModel(10,train_data.shape[1]-1,self.hidden)
                the_model.fit(train_data[:,:-1],train_data[:,-1])
                self.sector_models[sector] = the_model

        elif self.ensemble == "MLP":
            for sector in self.cluster_list:
                train_data, valid_data, _ = self.DM.data_phase(sector, self.phase, cluster=self.clustering)
                if withValidation: train_data = np.concatenate((train_data, valid_data), axis=0)
                a,b = train_data.shape[0], train_data.shape[1]
                train_data = train_data.reshape(a*b,-1)
                train_data = torch.Tensor(train_data).to(self.device)
                the_model = MultiLayerPerceptron(train_data.shape[1]-1,self.hidden,self.device)
                data = DataLoader(TensorDataset(train_data[:,:-1], train_data[:,-1]), batch_size=64, shuffle=True)
                the_model.fit(data,self.lr_MLP,self.epochs_MLP)
                self.sector_models[sector] = the_model
        else: #S3CE
            for sector in self.cluster_list:
                train_data, valid_data, _ = self.DM.data_phase(sector, self.phase, cluster=self.clustering)
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


        train_tmp, valid_tmp, _ = self.DM.data_phase("ALL",self.phase)
        if withValidation: train_tmp = np.concatenate((train_tmp, valid_tmp))
        train_data = train_tmp.reshape(train_tmp.shape[0]*train_tmp.shape[1],-1)
        # 전체 데이터를 가져와 train_data로 변환

        self.all_sector_model = AggregationModel(train_data.shape[1] - 1, self.n_rules, self.hidden, self.device)
        train_data = torch.Tensor(train_data).to(self.device)
        self.all_sector_model.fit(train_data[:, :-1], train_data[:, -1], self.epochs_MLP, self.epochs_anfis, self.lr_MLP,
                                  self.lr_anfis)
        # 개별 섹터 모델과 비교하기 위해 전체 시장을 학습한 모델을 실험

    def copymodels(self):
        for sector in self.DM.sector_list:
            self.sector_models[sector] = deepcopy(self.all_sector_model)

    def save_models(self,dir):
        joblib.dump(self,f"{dir}/model.joblib")

    def backtest(self, verbose=True, use_all='SectorAll', agg='inter', inter_n=0.1,withValidation = False, isTest=True, testNum=0, dir=""):  # 백테스팅 수행
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
            _, _, data_tmp = self.DM.data_phase(sector, self.phase, cluster=self.clustering)
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

            if use_all == "Sector" or use_all == "SectorAll":  # SectorAll or Sector일 경우
                for sector in self.cluster_list:  # 선택된 섹터별로 종목을 추천

                    model = self.sector_models[sector]
                    topK = model.predict(torch.Tensor(test_data[sector]).to(self.device)[i, :, :-1],
                                         symbols[sector])

                    if use_all == "SectorAll":  # SectorAll 모드: 전체 섹터 모델 활용, 전체 섹터 데이터를 기반으로 종목을 추천
                        self.all_sector_model.anfis = self.all_sector_model.anfis.type(torch.float64)
                        model = self.all_sector_model
                        if not isinstance(all_data, torch.Tensor):
                            all_data = torch.Tensor(all_data).to(self.device)
                        top_all = model.predict(all_data[i, :, :-1], all_symbol["code"])

                        inter_symbol = top_all.index.intersection(topK.index)  # 각 섹터의 모델 예측값과 전체 섹터 모델의 예측값을 조합

                        if agg == 'avg':
                            stocks = pd.concat([stocks, topK + top_all[inter_symbol]])
                        elif agg == 'inter':
                            if inter_n < 1:
                                cnt = len(topK)
                                topK = topK[:int(cnt * inter_n)]
                                top_all = top_all[inter_symbol][:int(cnt * inter_n)]
                            else:
                                topK = topK[:inter_n]
                                top_all = top_all[inter_symbol][:inter_n]

                            inter_stocks = top_all.index.intersection(topK.index).to_list()
                            real_last_topK_stock.extend(inter_stocks)

                    elif use_all == "Sector":  # Sector인 경우
                        real_last_topK_stock.extend(topK.index[:2].to_list())  # 상위 2개의 종목을 선택

            elif use_all == "All":  # 전체 데이터 기반 종목 선택
                model = self.all_sector_model
                if not isinstance(all_data, torch.Tensor):
                    all_data = torch.Tensor(all_data).to(self.device)
                top_all = model.predict(all_data[i, :, :-1], all_symbol["code"])

                real_last_topK_stock.extend(top_all.index[:self.final_stock_k].to_list())  # 최종 상위 k 개의 주식을 선택 및 추가

            if use_all == "SectorAll" and agg == 'avg':  # SectorAll 모드에서
                real_last_topK_stock = stocks.sort_values(ascending=False).index.to_list()[:self.final_stock_k]
                # 개별 섹터와 전체 섹터 모델의 예측값 평균을 사용하여 종목을 결정
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

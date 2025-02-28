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
            #if epoch % 10 == 0:
             #   print(f"\nMLP's Loss(Epoch {epoch}): {loss:.4f}")

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
                 ensemble="", clustering=False):
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

        self.epochs_MLP = 200  # MLP 학습 epoch
        self.lr_MLP = 0.001  # MLP 학습 rate
        self.hidden = 128  # MLP 은닉층 크기

        self.epochs_anfis = 200
        self.lr_anfis = 0.01
        self.n_rules = 10

        self.device = device

        self.DM = DataManager(feature_n)  # 데이터 관리 클래스
        self.Utils = Utils()  # 수익률 계산 등 유틸리티 함수 제공

        self.valid_models = {}
        self.sector_models = {}

    def trainSectorModels(self): # 섹터별 모델 학습
        # 개별 섹터 학습 모델
        if not self.clustering:
            sector_list = self.DM.sector_list
        else:
            sector_list = self.DM.cluster_list
        # 클러스터링 여부에 따라 알맞은 리스트 불러옴

        if self.ensemble == "bagging":
            for sector in sector_list:
                train_data,_,_ = self.DM.data_phase(sector,self.phase,cluster=self.clustering)
                a,b = train_data.shape[0], train_data.shape[1]
                train_data = train_data.reshape(a*b,-1)
                train_data = torch.Tensor(train_data).to(self.device)
                the_model = BaggingModel(10,train_data.shape[1]-1,128)
                the_model.fit(train_data[:,:-1],train_data[:,-1])
                self.sector_models[sector] = the_model

        elif self.ensemble == "MLP":
            for sector in sector_list:
                train_data,_,_ = self.DM.data_phase(sector,self.phase,cluster=self.clustering)
                a,b = train_data.shape[0], train_data.shape[1]
                train_data = train_data.reshape(a*b,-1)
                train_data = torch.Tensor(train_data).to(self.device)
                the_model = MultiLayerPerceptron(train_data.shape[1]-1,128,self.device)
                data = DataLoader(TensorDataset(train_data[:,:-1], train_data[:,-1]), batch_size=64, shuffle=True)
                the_model.fit(data,self.lr_MLP,self.epochs_MLP)
                self.sector_models[sector] = the_model

        elif self.ensemble == "RF":
            for sector in sector_list:
                train_data,_,_ = self.DM.data_phase(sector,self.phase,cluster=self.clustering)
                a,b = train_data.shape[0], train_data.shape[1]
                train_data = train_data.reshape(a*b,-1)
                the_model = RandomForestRegressor()
                the_model.fit(train_data[:,:-1], train_data[:,-1])
                self.sector_models[sector] = the_model

        else: # Aggregation 모델
            for sector in sector_list:
                train_data, _,_ = self.DM.data_phase(sector,self.phase,cluster=self.clustering)
                a,b = train_data.shape[0], train_data.shape[1]
                train_data = train_data.reshape(a*b,-1)
                train_data = torch.Tensor(train_data).to(self.device)
                the_model = AggregationModel(train_data.shape[1]-1,self.n_rules,self.hidden,self.device)
                the_model.fit(train_data[:,:-1],train_data[:,-1],self.epochs_MLP,self.epochs_anfis,self.lr_MLP,self.lr_anfis)

                self.sector_models[sector] = the_model

    def trainSectorModelsWithValid(self): # 훈련 데이터뿐만 아니라 검증 데이터도 포함하여 학습
        if self.ensemble == "bagging":
            for sector in self.topK_sectors:
                train_data,valid_data,_ = self.DM.data_phase(sector,self.phase,cluster=self.clustering)
                train_data = np.concatenate((train_data,valid_data),axis=0)
                a,b = train_data.shape[0], train_data.shape[1]
                train_data = train_data.reshape(a*b,-1)
                the_model = BaggingRegressor(MLPRegressor(128),10,max_samples=0.7)
                the_model.fit(train_data[:,:-1],train_data[:,-1])
                self.sector_models[sector] = the_model

        elif self.ensemble == "MLP":
            for sector in self.topK_sectors:
                train_data,valid_data,_ = self.DM.data_phase(sector,self.phase,cluster=self.clustering)
                train_data = np.concatenate((train_data,valid_data),axis=0)
                a,b = train_data.shape[0], train_data.shape[1]
                train_data = train_data.reshape(a*b,-1)
                train_data = torch.Tensor(train_data).to(self.device)
                the_model = MultiLayerPerceptron(train_data.shape[1]-1,128,self.device)
                data = DataLoader(TensorDataset(train_data[:,:-1], train_data[:,-1]), batch_size=64, shuffle=True)
                the_model.fit(data,self.lr_MLP,self.epochs_MLP)
                self.sector_models[sector] = the_model

        elif self.ensemble == "RF":
            for sector in self.topK_sectors:
                train_data,valid_data,_ = self.DM.data_phase(sector,self.phase,cluster=self.clustering)
                train_data = np.concatenate((train_data,valid_data),axis=0)
                a,b = train_data.shape[0], train_data.shape[1]
                train_data = train_data.reshape(a*b,-1)
                the_model = RandomForestRegressor()
                the_model.fit(train_data[:,:-1], train_data[:,-1])
                self.sector_models[sector] = the_model

        else:
            for sector in self.topK_sectors:
                train_data, valid_data,_ = self.DM.data_phase(sector,self.phase,cluster=self.clustering)
                train_data = np.concatenate((train_data,valid_data),axis=0)
                a,b = train_data.shape[0], train_data.shape[1]
                train_data = train_data.reshape(a*b,-1)
                train_data = torch.Tensor(train_data,).to(self.device)
                the_model = AggregationModel(train_data.shape[1]-1,self.n_rules,self.hidden,self.device)
                the_model.fit(train_data[:,:-1],train_data[:,-1],self.epochs_MLP,self.epochs_anfis,self.lr_MLP,self.lr_anfis)

                self.sector_models[sector] = the_model

    def trainALLSectorModel(self): # 전체 섹터를 하나의 모델로 학습
        # 전체 섹터 학습 모델
        train_data = np.ndarray([0])
        train_tmp, valid_tmp, _ = self.DM.data_phase("ALL",self.phase)
        train_tmp = np.concatenate((train_tmp, valid_tmp))
        train_data = train_tmp.reshape(train_tmp.shape[0]*train_tmp.shape[1],-1)
        # 전체 데이터를 가져와 train_data로 변환

        if self.ensemble == 'bagging':
            train_data = torch.Tensor(train_data).to(self.device)
            self.all_sector_model = BaggingModel(10,train_data.shape[1]-1,128)
            self.all_sector_model.fit(train_data[:,:-1],train_data[:,-1])

        elif self.ensemble == "MLP":
            train_data = torch.Tensor(train_data).to(self.device)
            self.all_sector_model = MultiLayerPerceptron(train_data.shape[1]-1,128,self.device)
            data = DataLoader(TensorDataset(train_data[:,:-1], train_data[:,-1]), batch_size=64, shuffle=True)
            self.all_sector_model.fit(data,self.lr_MLP,self.epochs_MLP)

        elif self.ensemble == "RF":
            self.all_sector_model = RandomForestRegressor()
            self.all_sector_model.fit(train_data[:,:-1], train_data[:,-1])

        else:
            self.all_sector_model = AggregationModel(train_data.shape[1]-1,self.n_rules,self.hidden,self.device)
            train_data = torch.Tensor(train_data).to(self.device)
            self.all_sector_model.fit(train_data[:,:-1],train_data[:,-1],self.epochs_MLP,200,self.lr_MLP,self.lr_anfis)
        # 개별 섹터 모델과 비교하기 위해 전체 시장을 학습한 모델을 실험

    def copymodels(self):
        for sector in self.DM.sector_list:
            self.sector_models[sector] = deepcopy(self.all_sector_model)

    def save_models(self,dir):
        joblib.dump(self,f"{dir}/model.joblib")

    def sectorModelsValidation(self):  # 검증을 통해 상위 섹터 선택
        # 섹터 모델 검증 및 상위 섹터 선택
        """return topK sector in validation period"""
        if not self.clustering:
            sector_list = self.DM.sector_list
        else:
            sector_list = self.DM.cluster_list

        valid_start = self.DM.phase_list[self.phase][1]
        valid_end = self.DM.phase_list[self.phase][2]
        sector_return = {}
        sector_loss = {}
        # valid_start~end 사이의 데이터로 검증 진행

        for sector in sector_list:
            symbol_index = pd.read_csv(f"./data/{sector}/symbol_index.csv", index_col=[0])["symbol"]
            _, valid_data, _ = self.DM.data_phase(sector, self.phase, cluster=self.clustering)
            model = self.sector_models[sector]

            if self.ensemble == "AGG3" or self.ensemble == "MLP":
                valid_data = torch.Tensor(valid_data).to(self.device)

            for pno in range(valid_start, valid_end):
                i = pno - valid_start

                if self.ensemble == "RF":
                    pred = model.predict(valid_data[i, :, :-1])
                    loss = mean_squared_error(pred, valid_data[i, :, -1].squeeze())
                elif self.ensemble == "MLP":
                    pred = model(valid_data[i, :, :-1]).cpu().detach().numpy().squeeze()
                    loss = mean_squared_error(pred, valid_data[i, :, -1].cpu().detach().numpy().squeeze())

                else:
                    loss = model.get_loss(valid_data[i, :, :-1], valid_data[i, :, -1], False)

                sector_loss[sector] = loss
                # get_loss를 통해 각 섹터 모델의 MSE를 계산

        self.topK_sectors = pd.Series(sector_loss).sort_values(ascending=True).index.to_list()[:self.valid_sector_k]
        # 손실이 가장 낮은 섹터를 self.topK_sectors에 저장

    def backtest(self, verbose=True, use_all='SectorAll', agg='avg', inter_n=5, isTest=True, testNum=0, dir=""):  # 백테스팅 수행
        # 선택된 섹터 및 전체 섹터 모델을 활용해 종목을 선택하고, 실제 데이터로 수익률을 평가
        # 과거 데이터를 사용하여 모델의 예측이 실제 시장에서 얼마나 잘 맞았는지를 검증하는 과정
        test_start = self.DM.phase_list[self.phase][2]
        test_end = self.DM.phase_list[self.phase][3]  # 테스트 데이터의 인덱스를 가져옴
        # 테스트 시작과 종료 시점 설정
        test_data = {}  # 섹터별 테스트 데이터를 저장할 딕셔너리
        symbols = {}  # 각 섹터별 종목(Symbol) 정보를 저장할 딕셔너리

        idx = 0
        clustered_stocks_list = []

        if use_all == "SectorAll" or use_all == "All":  # 전체 데이터를 불러옴
            _, _, all_data = self.DM.data_phase("ALL", self.phase)
            all_symbol = pd.read_csv(f"./data/f10/ALL/symbol_index.csv", index_col=[0])  # 전체 섹터 데이터 가져옴

        for sector in self.topK_sectors:  # 저장된 상위 섹터별 데이터를 로드
            _, _, data_tmp = self.DM.data_phase(sector, self.phase, cluster=self.clustering)
            test_data[sector] = data_tmp

            symbol_index = pd.read_csv(f"./data/{sector}/symbol_index.csv", index_col=[0])  # 해당 섹터의 주식 종목 리스트 가져옴
            symbols[sector] = symbol_index["symbol"]

        ## 백테스팅 진행
        pf_mem = []  # 각 날짜별 포트폴리오 수익률 기록
        self.moneyHistory = [1]  # 투자 금액의 변화를 기록
        num_of_stock = []  # 매일 선택된 주식 개수 저장

        if verbose: print(f"------[{self.phase}]------")

        for pno in range(test_start, test_end):  # test_start ~ end 기간동안 매일 반복 실행
            # 각 날짜마다 주식을 선택하고 수익률을 계산
            i = pno - test_start
            strdate = self.DM.pno2date(pno)  # 현재 날짜와
            next_strdate = self.DM.pno2date(pno + 1)  # 다음 날짜 반환

            stocks = pd.Series()  # 모든 섹터에서 선택된 주식을 저장
            real_last_topK_stock = []  # 최종적으로 선택된 상위 주식 저장
            clustered_stocks = {}

            if use_all == "Sector" or use_all == "SectorAll":  # SectorAll or Sector일 경우
                for sector in self.topK_sectors:  # 선택된 섹터별로 종목을 추천

                    if self.ensemble == 'bagging':
                        topK = pd.Series(self.sector_models[sector].predict(test_data[sector][i, :, :-1]),
                                         symbols[sector]).sort_values(ascending=False)

                    elif self.ensemble == "MLP":
                        topK = pd.Series(self.sector_models[sector](torch.Tensor(test_data[sector][i, :, :-1]).to(
                            self.device)).cpu().detach().numpy().squeeze(), symbols[sector]).sort_values(
                            ascending=False)

                    elif self.ensemble == "RF":
                        topK = pd.Series(self.sector_models[sector].predict(test_data[sector][i, :, :-1]),
                                         symbols[sector]).sort_values(ascending=False)

                    else:
                        model = self.sector_models[sector]
                        topK = model.predict(torch.Tensor(test_data[sector]).to(self.device)[i, :, :-1],
                                             symbols[sector])

                    if use_all == "SectorAll":  # SectorAll 모드: 전체 섹터 모델 활용, 전체 섹터 데이터를 기반으로 종목을 추천
                        if self.ensemble == 'bagging':
                            top_all = pd.Series(self.all_sector_model.predict(all_data[i, :, :-1]),
                                                all_symbol["symbol"]).sort_values(ascending=False)
                        elif self.ensemble == "MLP":
                            if not isinstance(all_data, torch.Tensor):
                                all_data = torch.Tensor(all_data).to(self.device)
                            top_all = pd.Series(
                                self.all_sector_model(all_data[i, :, :-1]).cpu().detach().numpy().squeeze(),
                                all_symbol["symbol"]).sort_values(ascending=False)

                        elif self.ensemble == "RF":
                            top_all = pd.Series(self.all_sector_model.predict(all_data[i, :, :-1]),
                                                all_symbol["symbol"]).sort_values(ascending=False)

                        else:
                            self.all_sector_model.anfis = self.all_sector_model.anfis.type(torch.float64)
                            model = self.all_sector_model
                            if not isinstance(all_data, torch.Tensor):
                                all_data = torch.Tensor(all_data).to(self.device)
                            top_all = model.predict(all_data[i, :, :-1], all_symbol["symbol"])

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
                if self.ensemble == 'bagging':
                    top_all = pd.Series(self.all_sector_model.predict(all_data[i, :, :-1]),
                                        all_symbol["symbol"]).sort_values(ascending=False)
                elif self.ensemble == "MLP":
                    if not isinstance(all_data, torch.Tensor):
                        all_data = torch.Tensor(all_data).to(self.device)
                    top_all = pd.Series(self.all_sector_model(all_data[i, :, :-1]).cpu().detach().numpy().squeeze(),
                                        all_symbol["symbol"]).sort_values(ascending=False)
                elif self.ensemble == "RF":
                    top_all = pd.Series(self.all_sector_model.predict(all_data[i, :, :-1]),
                                        all_symbol["symbol"]).sort_values(ascending=False)
                else:
                    model = self.all_sector_model
                    if not isinstance(all_data, torch.Tensor):
                        all_data = torch.Tensor(all_data).to(self.device)
                    top_all = model.predict(all_data[i, :, :-1], all_symbol["symbol"])

                real_last_topK_stock.extend(top_all.index[:self.final_stock_k].to_list())  # 최종 상위 k 개의 주식을 선택 및 추가

            if use_all == "SectorAll" and agg == 'avg':  # SectorAll 모드에서
                real_last_topK_stock = stocks.sort_values(ascending=False).index.to_list()[:self.final_stock_k]
                # 개별 섹터와 전체 섹터 모델의 예측값 평균을 사용하여 종목을 결정
            clustered_stocks_list.append([f"Cluster {idx}"] + real_last_topK_stock)
            idx += 1
            self.final_stock_k = len(real_last_topK_stock)  # 최종적으로 선택된 주식 개수를 저장

            if verbose: print(real_last_topK_stock)  # 선택된 최종 종목을 출력
            if isTest:
                pd.DataFrame(clustered_stocks_list).to_csv(f"./result/{dir}/test_selected_stocks_{self.phase}_{testNum}.csv", index=False)
            if not isTest:
                pd.DataFrame(clustered_stocks_list).to_csv(
                    f"{dir}/train_selected_stocks_{self.phase}.csv", index=False)
            num_of_stock.append(len(real_last_topK_stock))

            # 실제 데이터 로딩 및 포트폴리오 성과 평가
            fs = pd.read_csv(f"./data/date_regression/{strdate}.csv")  # 현재 날짜의 주가 데이터
            if next_strdate != '2024-06-30':
                next_fs = pd.read_csv(f"./data/date_regression/{next_strdate}.csv")  # 다음 날짜의 주가 데이터

            daily_change = self.Utils.get_portfolio_memory(real_last_topK_stock, strdate, next_strdate, fs, next_fs)
            # 매일의 포트폴리오 수익률 계산
            # 선택된 주식 리스트, 현재 및 다음 날짜의 주가 데이터 이용

            pf_mem.extend(daily_change)  # 매일 수익률을 리스트에 추가: 각 날짜별 수익률 저장

        return_ratio = np.prod(np.array(pf_mem) + 1) - 1
        mdd = self.Utils.get_MDD(np.array(pf_mem) + 1)
        sharpe = self.Utils.get_sharpe_ratio(pf_mem)

        if verbose:
            print(f"MDD: {mdd}, Sharpe: {sharpe}, CAGR: {return_ratio}")
            print("----------------------")

        return return_ratio, sharpe, mdd, num_of_stock

    def save(obj):
        return (obj.__class__, obj.__dict__)

    def load(cls, attributes):
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

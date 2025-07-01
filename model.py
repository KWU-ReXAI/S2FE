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
        self.all_sector_model.fit(train_data[:, :-1], train_data[:, -1], self.epochs_MLP, self.epochs_anfis, self.lr_MLP,
                                  self.lr_anfis)
        # 개별 섹터 모델과 비교하기 위해 전체 시장을 학습한 모델을 실험

    def copymodels(self):
        for sector in self.DM.sector_list:
            self.sector_models[sector] = deepcopy(self.all_sector_model)

    def save_models(self,dir):
        joblib.dump(self,f"{dir}/model.joblib")
    
    def get_rows_by_date_range(self, code: str, start_date_str: str, end_date_str: str, data: int=0) -> pd.DataFrame:
        # 실제 존재하는 파일 경로를 찾기
        file_path = None
        if data == 0: file_path = "./preprocessed_data/llm/predict_video/predict.csv"
        elif data == 1: file_path = "./preprocessed_data/llm/predict_text/predict.csv"
        elif data == 2: file_path = "./preprocessed_data/llm/predict_mix/predict.csv"
        else: raise ValueError('data 파라미터가 범위를 벗어남.')
        
        # CSV 읽기
        df = pd.read_csv(file_path, encoding='utf-8')
        df = df[df['code'] == code]

        # upload_dt를 datetime으로 변환
        df['upload_dt'] = pd.to_datetime(df['upload_dt'])

        # 문자열을 datetime으로 변환
        start_dt = pd.to_datetime(start_date_str) - relativedelta(months=1)
        end_dt = pd.to_datetime(end_date_str) - relativedelta(months=1)

        # 업로드날짜가 기간 내에 존재할 경우
        mask = (start_dt <= df['upload_dt']) & (df['upload_dt'] < end_dt)

        # 겹치는 행 반환
        return df.loc[mask].reset_index(drop=True)
    
    def LLM_task(self, model_list, start_date, end_date, data, initial_balance):
        balance = initial_balance
        daily_pv = []
        # 1) model_list를 index 리스트로 변환
        if model_list is None: return None, None, None, None

        # 2) 해당 분기의 가장 늦는 공시일 얻기
        start_datetime = self.DM.get_disclosure_date(start_date)
        end_datetime = self.DM.get_disclosure_date(end_date)
        
        # 3) 개별 코드별로 DataFrame 모아두기
        df_list = []
        if data != 3:
            for code in model_list:
                print(f"{code} of {start_date}~{end_date} : {start_datetime}~{end_datetime}")
                df = self.get_rows_by_date_range(code, start_datetime, end_datetime, data)
                if df.empty:
                    continue
                df_list.append(df)


        start_dt = datetime.strptime(start_datetime, '%Y-%m-%d') + timedelta(days=1) # 점검 필요
        end_dt = datetime.strptime(end_datetime, '%Y-%m-%d') - timedelta(days=1)
        current = start_dt ## 거래일

        ### 월 단위 거래 수익률(1달 뒤 자산 / 현재 자산 - 1)
        month_return = []
        ### 월 단위로 거래된 주식 종목 리스트: [[삼성전자, LG], [LG, HMM], [SK하이닉스, HMM]]
        month_stock_list = []
        ### 거래 일자 저장 리스트
        month_trade_dates = []
        while current <= end_dt:

            month_trade_dates.append(current.strftime("%Y-%m-%d"))

            if data == 3:
                # 데이터프레임 생성을 위한 딕셔너리 구성
                tmp = {
                    'code': model_list,
                    'score': None
                }
                # 딕셔너리를 사용하여 데이터프레임 생성
                df_select = pd.DataFrame(tmp)
            else:
                # df_select는 한 달 주기로 뽑히는 종목들
                df_select = pd.DataFrame(columns=[
                    "code", "score"
                ])
            for df_now in df_list:
                df_now['upload_dt'] = pd.to_datetime(df_now['upload_dt'])

                ### 자료 수집 기간: upload_start ~ upload_end ###
                upload_start = current - relativedelta(months=1)
                upload_end = current - timedelta(days=1)
                chunk = df_now.loc[(upload_start <= df_now['upload_dt']) & (df_now['upload_dt'] <= upload_end)].copy()

                chunk.reset_index(drop=True, inplace=True)
                chunk["score"] *= (chunk.index + 1) #가중치 넣음! 우선 임의로

                # score 합계 및 code 추출
                score_sum = chunk["score"].sum()
                code_val = df_now["code"].iloc[0]

                # 한 행으로 추가
                df_select.loc[len(df_select)] = [
                    code_val, score_sum
                ]
            ### threshold는 상의 후 결정하기
            threshold = 2

            if data != 3:
                # 종목 거르기 및 걸러진 종목 저장
                df_select = df_select[df_select['score'] >= threshold]
            month_stock_list.append(df_select['code'].tolist())

            # 잔액을 LLM에 의해 걸러진 종목 개수로 나눔(월별로 다름)
            num_stock = len(df_select)
            balance_divided = balance // num_stock if num_stock != 0 else balance
            balance_remainder = balance % num_stock if num_stock != 0 else 0

            ###### 구매 코드 구현하기 ######
            buy_dt = current
            sell_dt = current + relativedelta(months=1)
            sell_dt = sell_dt if sell_dt <= end_dt else end_dt

            charge = 0.005

            # LLM에 의해 걸러진 종목만 거래
            prev_balance = balance # 월별 수익률 계산 위함
            monthly_pv = []
            if df_select.empty:
                code = str(int(model_list[0])).zfill(6)
                df_price = pd.read_csv(f"data_kr/price/{code}.csv")
                df_price['날짜'] = pd.to_datetime(df_price['날짜'])
                prices = df_price[(df_price['날짜'] >= buy_dt) & (df_price['날짜'] <= sell_dt)]['종가'].tolist()
                result = [balance for x in range(len(prices))]
            else:
                for row in df_select.itertuples():
                    code = str(int(row.code)).zfill(6)
                    df_price = pd.read_csv(f"data_kr/price/{code}.csv")
                    df_price['날짜'] = pd.to_datetime(df_price['날짜'])
                    buy_price = df_price[df_price['날짜'] >= buy_dt].iloc[0]['종가']
                    sell_price = df_price[df_price['날짜'] <= sell_dt].iloc[-1]['종가']

                    ### 거래 (판매 수수료 0.5%) ###
                    num_stock = balance_divided // buy_price
                    balance -= buy_price * num_stock
                    balance += (sell_price * num_stock) * (1 - charge)

                    ### 일간 pv 변화 종목별로 기록
                    prices = df_price[(df_price['날짜'] >= buy_dt) & (df_price['날짜'] <= sell_dt)]['종가'].tolist()
                    prices = [x * num_stock + balance_divided % buy_price for x in prices]
                    prices[-1] *= (1 - charge)
                    monthly_pv.append(prices)
                # 모든 종목의 일간 pv 더하기 + 거래 잔액 추가
                result = [sum(elements) for elements in zip(*monthly_pv)]
                result = [x + balance_remainder for x in result]
            daily_pv.extend(result)
            month_return.append(balance / prev_balance - 1) # 월별 수익률: 이후 pv/이전 pv - 1
            current += relativedelta(months=1)
        return month_return, month_stock_list, month_trade_dates, balance, daily_pv

    def backtest(self, verbose=True, use_all='Sector', agg='inter', inter_n=0.1, withValidation = False, isTest=True, testNum=0, dir="", withLLM = False, LLMagg = "False"):  # 백테스팅 수행
        # 선택된 섹터 및 전체 섹터 모델을 활용해 종목을 선택하고, 실제 데이터로 수익률을 평가
        # 과거 데이터를 사용하여 모델의 예측이 실제 시장에서 얼마나 잘 맞았는지를 검증하는 과정
        test_start = self.DM.phase_list[self.phase][2 if withValidation else 1]
        test_end = self.DM.phase_list[self.phase][3 if withValidation else 2]
        # 테스트 시작과 종료 시점 설정
        test_data = {}  # 섹터별 테스트 데이터를 저장할 딕셔너리
        symbols = {}  # 각 섹터별 종목(Symbol) 정보를 저장할 딕셔너리

        clustered_stocks_list = [[] for x in range(4)]

        if use_all == "SectorAll" or use_all == "All":  # 전체 데이터를 불러옴
            _, _, all_data = self.DM.data_phase("ALL", self.phase)
            all_symbol = pd.read_csv(f"./data_kr/symbol.csv")  # 전체 섹터 데이터 가져옴

        for sector in self.cluster_list:  # 저장된 상위 섹터별 데이터를 로드
            _, _, data_tmp = self.DM.data_phase(sector, self.phase, cluster=self.clustering, LLM = withLLM)
            test_data[sector] = data_tmp

            symbol_index = pd.read_csv(f"./preprocessed_data/{sector}/symbol_index.csv")  # 해당 섹터의 주식 종목 리스트 가져옴
            symbols[sector] = symbol_index["Code"]

        ## 백테스팅 진행
        pf_mem_ks = []
        num_of_stock = []  # 매일 선택된 주식 개수 저장

        pf_mem = [[] for x in range(4)]
        pf_mem_daily = [[] for x in range(4)] # 일 단위 MDD를 위한 메모리
        pf_mem_dates = []

        if verbose: print(f"\n------[{self.phase}]------",flush=True)

        print(f"Test Period: {self.DM.pno2date(test_start)} ~ {self.DM.pno2date(test_end-1)}")

        balance = [100000000.0 for i in range(4)]  # 1억 -> 분기별로 초기화되는 잔액을 위와 같이 수정

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

                selected = topK.index.to_list()
                real_last_topK_stock.extend(selected)

            ### real_last_topK_stock: 섹터별 상위 20% 종목들
            ### 0: video, 1: article, 2: mix, 3: model(without LLM)
            for idx, data in enumerate(['video', 'article', 'mix', 'model']):
                month_return, month_stock_lists, month_trade_dates, balance[idx], daily_pv = \
                    self.LLM_task(real_last_topK_stock, strdate, next_strdate, idx, balance[idx])

                index = 0
                if isTest:
                    for month_list in month_stock_lists:
                        clustered_stocks_list[idx].append([f"{strdate}", f"{month_trade_dates[index]}"] + month_list)
                        index += 1
                self.final_stock_k = len(real_last_topK_stock)  # 최종적으로 선택된 주식 개수를 저장

                if verbose: print(real_last_topK_stock,flush=True)  # 선택된 최종 종목을 출력
                if isTest:
                    pd.DataFrame(clustered_stocks_list[idx]).to_csv(f"./result/{dir}/test_selected_stocks_{data}_{self.phase}_{testNum}.csv", index=False)
                if not isTest:
                    pd.DataFrame(clustered_stocks_list[idx]).to_csv(
                        f"{dir}/train_selected_stocks_{self.phase}.csv", index=False)
                num_of_stock.append(len(real_last_topK_stock))

                ### 수익률, 거래일 저장
                pf_mem[idx].extend(month_return)
                pf_mem_daily[idx].extend(daily_pv)
                if data == 'model': pf_mem_dates.extend(month_trade_dates)

                daily_change = self.Utils.get_portfolio_memory(real_last_topK_stock, strdate, next_strdate,False)
                daily_change_KOSPI = self.Utils.get_portfolio_memory(real_last_topK_stock,strdate,next_strdate,True)
                # 매일의 포트폴리오 수익률 계산
                # 선택된 주식 리스트, 현재 및 다음 날짜의 주가 데이터 이용
                pf_mem.extend(daily_change)
                pf_mem_ks.extend(daily_change_KOSPI)

        if isTest:
            ### 0: video, 1: article, 2: mix, 3: model(without LLM)
            for idx, data in enumerate(['video', 'article', 'mix', 'model']):
                return_dict = {
                    'date': pf_mem_dates,
                    'return': pf_mem[idx]
                }
                df_return = pd.DataFrame(return_dict)
                df_return.to_csv(
                f"./result/{dir}/test_monthly_{data}_return_{self.phase}_{testNum}.csv", index=False)


        # LLM 필터링 안 한 모델의 평가지표
        return_ratio = np.prod(np.array(pf_mem[3]) + 1) - 1
        mdd = self.Utils.get_MDD(np.array(pf_mem_daily[3]) + 1)
        sharpe = self.Utils.get_sharpe_ratio(pf_mem[3])

        # 영상 평가지표
        return_ratio_video = np.prod(np.array(pf_mem[0]) + 1) - 1
        mdd_video = self.Utils.get_MDD(np.array(pf_mem_daily[0]) + 1)
        sharpe_video = self.Utils.get_sharpe_ratio(pf_mem[0])

        # 기사 평가지표
        return_ratio_article = np.prod(np.array(pf_mem[1]) + 1) - 1
        mdd_article= self.Utils.get_MDD(np.array(pf_mem_daily[1]) + 1)
        sharpe_article = self.Utils.get_sharpe_ratio(pf_mem[1])

        # 영상+기사 평가지표
        return_ratio_mix = np.prod(np.array(pf_mem[2]) + 1) - 1
        mdd_mix = self.Utils.get_MDD(np.array(pf_mem_daily[2]) + 1)
        sharpe_mix = self.Utils.get_sharpe_ratio(pf_mem[2])

        # KOSPI (출력 X, Train에서 에러 없기 위함)
        return_ratio_ks = np.prod(np.array(pf_mem_ks) + 1) - 1
        mdd_ks = self.Utils.get_MDD(np.array(pf_mem_ks) + 1)
        sharpe_ks = self.Utils.get_sharpe_ratio(pf_mem_ks)


        if verbose:
            print(f"\nMDD: {mdd}\tSharpe: {sharpe}\tCAGR: {return_ratio}",flush=True)
            print(f"Video MDD: {mdd_video}\tVideo Sharpe: {sharpe_video}\tVideo CAGR: {return_ratio_video}",flush=True)
            print(f"Article MDD: {mdd_article}\tArticle Sharpe: {sharpe_article}\tArticle CAGR: {return_ratio_article}", flush=True)
            print(f"Mix MDD: {mdd_mix}\tMix Sharpe: {sharpe_mix}\tMix CAGR: {return_ratio_mix}", flush=True)
            print("----------------------",flush=True)
        if isTest:
            return return_ratio, sharpe, mdd,num_of_stock, return_ratio_video, sharpe_video, mdd_video, \
                return_ratio_article, sharpe_article, mdd_article, return_ratio_mix, sharpe_mix, mdd_mix
        else:
            return return_ratio, sharpe, mdd, num_of_stock, return_ratio_ks, sharpe_ks, mdd_ks

    def save(obj):
        return (obj.__class__, obj.__dict__)

    def load(cls, attributes):
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

if __name__ == '__main__':
    testNum = 1
    cluster_n = 5
    DM = DataManager(features_n=6, cluster_n=cluster_n)  # 특징 개수 4개로 설정하여 데이터 매니저 초기화
    DM.create_date_list()
    phase_list = DM.phase_list.keys()

    for K in range(1, testNum + 1):
        for phase in phase_list:
            model = joblib.load(f"./result/train_result_dir_{K}/train_result_model_{K}_{phase}/model.joblib")  # 저장된 모델 불러옴
            cagr, sharpe, mdd, num_stock_tmp, cagr_video, sharpe_video, mdd_video, cagr_article, sharpe_article, mdd_article, cagr_mix, sharpe_mix, mdd_mix \
                = model.backtest(verbose=True, agg="inter", use_all="Sector", inter_n=0.2, withValidation=True,
                                 isTest=True, testNum=K, dir="test_result_dir", withLLM=False)
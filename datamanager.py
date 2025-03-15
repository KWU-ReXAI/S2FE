import pandas as pd
import numpy as np
from FinDataLoader import FinDataLoader as fd
from datetime import datetime
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class DataManager:
    def __init__(self, features_n):
        self.features_n = features_n
        self.ticker_list = sorted(pd.read_csv("./data_kr/symbol.csv")["code"].tolist())
        self.sector_list = list(set(pd.read_csv("./data_kr/symbol.csv")["sector"]))
        self.cluster_list = ["cluster_" + str(i) for i in range(3)]  # 클러스터 번호 부여

        self.phase_list = {"p1": [1, 15, 19, 23], "p2": [5, 19, 23, 27], "p3": [9, 23, 27, 31], "p4": [13, 27, 31, 35]}

    def create_date_list(self):
        # 예시: merged 폴더의 파일 이름이 "2015_Q1.csv", "2015_Q2.csv" 등이라면
        files = os.listdir("./data_kr/date_regression")
        dates = [f.replace(".csv", "") for f in files if f.endswith(".csv")]
        dates = sorted(dates)  # 날짜 순으로 정렬 (적절한 정렬 기준을 적용)
        self.date_list = dates
        return dates

    def get_sector_list(self):
        return self.sector_list
    def get_ticker_list(self):
        return self.ticker_list

    def date2quarter(date_str):
        year, month, _ = map(int, date_str.split("-"))  # "2023-07-15" → (2023, 7, 15)

        # 월에 따라 분기 결정
        if 1 <= month <= 3:
            quarter = "Q1"
        elif 4 <= month <= 6:
            quarter = "Q2"
        elif 7 <= month <= 9:
            quarter = "Q3"
        else:
            quarter = "Q4"

        return f"{year}_{quarter}"

    def quarter2date(quarter_str):
        year, q = quarter_str.split("_")  # "2018_Q3" → ["2018", "Q3"]
        year = int(year)

        # 분기별 시작 & 종료 날짜 설정
        quarter_dict = {
            "Q1": ("01-01", "03-31"),
            "Q2": ("04-01", "06-30"),
            "Q3": ("07-01", "09-30"),
            "Q4": ("10-01", "12-31"),
        }

        if q in quarter_dict:
            start_date = f"{year}-{quarter_dict[q][0]}"  # 시작일
            end_date = f"{year}-{quarter_dict[q][1]}"  # 종료일
            return start_date, end_date
        else:
            raise ValueError("Invalid quarter format. Use YYYY_QX (e.g., 2018_Q3)")

    def pno2date(self, pno: int) -> str:
        try:
            return self.date_list[pno]
        except IndexError:
            raise ValueError(f"pno {pno}은 date_list의 범위를 벗어났습니다. date_list 길이: {len(self.date_list)}")

    def data_phase(self, sector: str, phase: str, pandas_format=False, cluster=False, isall=False):
        train_start = self.phase_list[phase][0]
        valid_start = self.phase_list[phase][1]
        test_start = self.phase_list[phase][2]
        test_end = self.phase_list[phase][3]

        if isall:
            train_data = pd.DataFrame()
            valid_data = pd.DataFrame()
            test_data = pd.DataFrame()

            for pno in range(train_start, valid_start):
                strdate = self.pno2date(pno)  # 예: "2015_Q1"
                fp = f"./data_kr/financial_with_Label/{sector}/{strdate}.csv"
                fs_data = pd.read_csv(fp)
                fs_data.drop(["name", "sector", "year", "quarter", "Code"], axis=1, inplace=True)
                train_data = pd.concat([train_data, fs_data], axis=0)

            for pno in range(valid_start, test_start):
                strdate = self.pno2date(pno)
                fp = f"./data_kr/financial_with_Label/{sector}/{strdate}.csv"
                fs_data = pd.read_csv(fp)
                fs_data.drop(["name", "sector", "year", "quarter", "Code"], axis=1, inplace=True)
                valid_data = pd.concat([valid_data, fs_data], axis=0)

            for pno in range(test_start, test_end):
                strdate = self.pno2date(pno)
                fp = f"./data_kr/financial_with_Label/{sector}/{strdate}.csv"
                fs_data = pd.read_csv(fp)
                fs_data.drop(["name", "sector", "year", "quarter", "Code"], axis=1, inplace=True)
                test_data = pd.concat([test_data, fs_data], axis=0)

            return train_data, valid_data, test_data
        if pandas_format:
            train_data = pd.DataFrame()
            valid_data = pd.DataFrame()
            test_data = pd.DataFrame()

            for pno in range(train_start, valid_start):
                strdate = self.pno2date(pno)  # 예: "2015_Q1"
                fp = f"./data_kr/date_sector/{sector}/{strdate}.csv"
                fs_data = pd.read_csv(fp)
                # fs_data.drop(["Date", "symbol", "Filing Date"], axis=1, inplace=True)
                train_data = pd.concat([train_data, fs_data], axis=0)

            for pno in range(valid_start, test_start):
                strdate = self.pno2date(pno)
                fp = f"./data_kr/date_sector/{sector}/{strdate}.csv"
                fs_data = pd.read_csv(fp)
                # fs_data.drop(["Date", "symbol", "Filing Date"], axis=1, inplace=True)
                valid_data = pd.concat([valid_data, fs_data], axis=0)

            for pno in range(test_start, test_end):
                strdate = self.pno2date(pno)
                fp = f"./data_kr/date_sector/{sector}/{strdate}.csv"
                fs_data = pd.read_csv(fp)
                # fs_data.drop(["Date", "symbol", "Filing Date"], axis=1, inplace=True)
                test_data = pd.concat([test_data, fs_data], axis=0)

            return train_data, valid_data, test_data

        # cluster 옵션: 클러스터링된 데이터를 사용 (경로에 _cluster 추가)
        if cluster:
            train_list = []
            valid_list = []
            test_list = []

            for pno in range(train_start, valid_start):
                strdate = self.pno2date(pno)
                fs_data = pd.read_csv(f"./data_kr/clustered_data/{sector}/{strdate}.csv",index_col=[0])
                fs_data.drop(["name", "sector", "year", "quarter", "Code"], axis=1, inplace=True)
                train_list.append(fs_data)

            for pno in range(valid_start, test_start):
                strdate = self.pno2date(pno)
                fs_data = pd.read_csv(f"./data_kr/clustered_data/{sector}/{strdate}.csv", index_col=[0])
                fs_data.drop(["name", "sector", "year", "quarter", "Code"], axis=1, inplace=True)
                valid_list.append(fs_data)

            for pno in range(test_start, test_end):
                strdate = self.pno2date(pno)
                fs_data = pd.read_csv(f"./data_kr/clustered_data/{sector}/{strdate}.csv", index_col=[0])
                fs_data.drop(["name", "sector", "year", "quarter", "Code"], axis=1, inplace=True)
                test_list.append(fs_data)

            return np.array(train_list), np.array(valid_list), np.array(test_list)

        # 기본: numpy 배열 형태로 반환 (cluster=False, pandas_format=False)
        train_data = []
        valid_data = []
        test_data = []

        for pno in range(train_start, valid_start):
            strdate = self.pno2date(pno)  # 예: "2015_Q1"
            fs_data = pd.read_csv(f"./data_kr/clustered_data/{sector}/{strdate}.csv",index_col=[0])
            fs_data.drop(["name", "sector", "year", "quarter", "Code"], axis=1, inplace=True)
            train_data.append(fs_data)

        for pno in range(valid_start, test_start):
            strdate = self.pno2date(pno)
            fs_data = pd.read_csv(f"./data_kr/clustered_data/{sector}/{strdate}.csv",index_col=[0])
            fs_data.drop(["name", "sector", "year", "quarter", "Code"], axis=1, inplace=True)
            valid_data.append(fs_data)

        for pno in range(test_start, test_end):
            strdate = self.pno2date(pno)
            fs_data = pd.read_csv(f"./data_kr/clustered_data/{sector}/{strdate}.csv",index_col=[0])
            fs_data.drop(["name", "sector", "year", "quarter", "Code"], axis=1, inplace=True)
            test_data.append(fs_data)

        train_data = np.array(train_data)
        valid_data = np.array(valid_data)
        test_data = np.array(test_data)

        return train_data, valid_data, test_data



